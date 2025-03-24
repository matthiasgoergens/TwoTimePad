import os.path
import numpy as np
import os
import random

# Monkey patching to make np.inf work with TensorFlow.
np.Inf = np.inf

import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Add,
    Average,
    BatchNormalization,
    LayerNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GaussianDropout,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    Layer,
    MaxPooling1D,
    PReLU,
    SeparableConv1D,
    SimpleRNN,
    Softmax,
    SpatialDropout1D,
    TimeDistributed,
    average,
    concatenate,
    Concatenate,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model, Sequential

from ops import avg, cat, concat, plus
from process_corpus import alpha


def setup():
    np.set_printoptions(precision=4)

    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        useGPU = False
        print(SystemError("GPU device not found", device_name))
        raise NotImplementedError("Want GPU")
    else:
        useGPU = True
        print("Found GPU at: {}".format(device_name))

    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


class LastCharLoss(tf.keras.metrics.Mean):
    def __init__(self, name="last_char_loss", **kwargs):
        super(LastCharLoss, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(
            y_true[-1], y_pred[-1], from_logits=True
        )
        return super(LastCharLoss, self).update_state(loss_value, sample_weight)


window_size = 100
batch_size = 256
subset_size = 100_000

corpus_filename = "corpus.bytes"


class RandomSubsetSequence(tf.keras.utils.Sequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get total size in bytes of the corpus.
        self.total_bytes = os.path.getsize(corpus_filename)
        # Each sample requires window_size + 1 bytes.
        self.total_samples = self.total_bytes - window_size

        # Build the dataset for the first epoch.
        self.build_dataset()

    def build_dataset(self):
        # Choose a random offset (header_bytes) so that there are enough samples left.
        max_header = max(self.total_bytes - window_size * (subset_size + 1), 0)
        header_bytes = random.randrange(max_header)

        # Create the dataset that starts reading after header_bytes.
        dataset = tf.data.FixedLengthRecordDataset(
            corpus_filename, record_bytes=1, header_bytes=header_bytes
        )

        # Decode each record (byte) into a uint8.
        dataset = dataset.map(lambda x: tf.io.decode_raw(x, tf.uint8)[0])

        # Create sliding windows of window_size+1 so that each window gives you
        # an input (first window_size bytes) and target (bytes shifted by one).
        windowed_dataset = dataset.window(
            window_size + 1, shift=window_size, drop_remainder=True
        )
        windowed_dataset = windowed_dataset.flat_map(
            lambda window: window.batch(window_size + 1)
        )

        # Split each window into (input, target)
        def split_input_target(window):
            return window[:window_size], window[1:]

        dataset = windowed_dataset.map(split_input_target)

        # Take only a contiguous block (subset) for the current epoch.
        dataset = dataset.take(subset_size)

        # (Optional) Shuffle within this subset if desired.
        dataset = dataset.shuffle(
            buffer_size=subset_size, reshuffle_each_iteration=True
        )

        # Batch and prefetch for performance.
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # Materialize the dataset into a list of batches (this is fine if subset_size is moderate).
        self.dataset_batches = list(dataset.as_numpy_iterator())

    def __len__(self):
        # Returns the number of batches per epoch.
        return len(self.dataset_batches)

    def __getitem__(self, idx):
        return self.dataset_batches[idx]

    def on_epoch_end(self):
        # At the end of each epoch, rebuild the dataset with a new random header offset.
        self.build_dataset()


def make_data():
    # Read the file one byte at a time.
    dataset = tf.data.FixedLengthRecordDataset(corpus_filename, record_bytes=1)

    # Decode the raw bytes into uint8 values.
    dataset = dataset.map(lambda x: tf.io.decode_raw(x, tf.uint8)[0])

    # Create sliding windows of window_size+1 (to have enough for both input and target)
    windowed_dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    windowed_dataset = windowed_dataset.flat_map(
        lambda window: window.batch(window_size + 1)
    )

    # Split each window into (input, target)
    # Input: first window_size bytes
    # Target: last window_size bytes (shifted by 1 from input)
    def split_input_target(window):
        input_bytes = window[:window_size]  # First window_size bytes
        target_bytes = window[1:]  # Last window_size bytes (shifted by 1)
        return input_bytes, target_bytes

    dataset = windowed_dataset.map(split_input_target)

    # (Optional) Shuffle and batch the dataset.
    dataset = (
        dataset.shuffle(1_000_000)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


def make_model():
    relu = PReLU
    embedded_output_dim = 64

    flatten = Lambda(
        lambda x: tf.reshape(x, [tf.shape(x)[0], window_size * embedded_output_dim])
    )

    inputs = Input(shape=(window_size,))
    # 46 unique tokens mapped to 64-dim embeddings
    outputs = Sequential(
        [
            Embedding(input_dim=46, output_dim=embedded_output_dim),
            flatten,
        ]
    )(inputs)

    # outputs = Flatten()(embedded)
    height = 10
    blowup = 256
    for i in range(height):
        outputs = cat(
            outputs,
            Sequential(
                [
                    BatchNormalization(),
                    relu(),
                    Dense(blowup),
                ]
            )(outputs),
        )
    outputs = Dense(len(alpha))(outputs)
    model = model = Model([inputs], [outputs])
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    checkpoint_dir = "checkpoints/cumulative1"
    return (model, checkpoint_dir)


def make_model_gru():
    embedded_output_dim = 46

    model = Sequential(
        [
            Input(shape=(window_size,)),
            Embedding(input_dim=len(alpha), output_dim=embedded_output_dim),
            LayerNormalization(),
            GRU(256, return_sequences=True),
            LayerNormalization(),
            GRU(256, return_sequences=True),
            LayerNormalization(),
            GRU(256),
            LayerNormalization(),
            Dense(len(alpha)),
        ]
    )

    model.compile(
        optimizer=tf.optimizers.Adam(global_clipnorm=1.0),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "checkpoints/gru_ln3_twice"
    return (model, checkpoint_dir)


def make_model_gru_skip():
    embedded_output_dim = 46

    # Layer configuration - all return sequences now
    layer_config = [
        {"units": 256, "skip_from": [], "skip_type": None},
        {"units": 256, "skip_from": [0], "skip_type": "residual"},
        {"units": 256, "skip_from": [0, 1], "skip_type": "concat"},
        {"units": 256, "skip_from": [0, 1, 2], "skip_type": "concat"},
        {"units": 256, "skip_from": [0, 1, 2, 3], "skip_type": "concat"},
        {"units": 256, "skip_from": [0, 1, 2, 3, 4], "skip_type": "concat"},
    ]

    # Input layer
    inputs = Input(shape=(window_size,))

    # Embedding layer
    x = Embedding(input_dim=len(alpha), output_dim=embedded_output_dim)(inputs)
    embed = LayerNormalization()(x)

    # Store layer outputs for skip connections
    layer_outputs = [embed]  # Start with embedding as first layer output

    # Create GRU layers with skip connections
    for i, config in enumerate(layer_config):
        current_input = layer_outputs[-1]

        # Handle skip connections
        if config["skip_type"] == "residual" and config["skip_from"]:
            # For residual connections, project to match dimensions
            skip_sources = [
                TimeDistributed(Dense(config["units"]))(layer_outputs[j])
                for j in config["skip_from"]
            ]
            for skip in skip_sources:
                current_input = Add()([current_input, skip])

        elif config["skip_type"] == "concat" and config["skip_from"]:
            # For concat connections, concatenate then project
            skip_sources = [layer_outputs[j] for j in config["skip_from"]]
            concat = Concatenate()([current_input] + skip_sources)
            current_input = TimeDistributed(Dense(config["units"]))(concat)

        # Create GRU layer - always return sequences
        gru_output = GRU(config["units"], return_sequences=True, name=f"gru_{i}")(
            current_input
        )

        # Normalize output
        norm_output = LayerNormalization()(gru_output)

        # Store sequence output for skip connections
        layer_outputs.append(norm_output)

    # Concatenate all layer outputs along the feature dimension
    final_concat = Concatenate()(layer_outputs)

    # Output layer - predict at each timestep
    outputs = TimeDistributed(Dense(len(alpha)))(final_concat)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(global_clipnorm=1.0),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", LastCharLoss()],
    )

    model.summary()
    checkpoint_dir = "checkpoints/gru_to_final_sequence_5"
    return (model, checkpoint_dir)


def make_model_relu_rnn():
    embedded_output_dim = 64

    model = tf.keras.Sequential(
        [
            Input(shape=(window_size,)),
            Embedding(input_dim=len(alpha), output_dim=embedded_output_dim),
            BatchNormalization(),
            # First SimpleRNN layer with ReLU
            SimpleRNN(768, activation="linear", return_sequences=True),
            PReLU(),
            BatchNormalization(),
            # Second SimpleRNN layer with ReLU
            SimpleRNN(768, activation="linear", return_sequences=True),
            PReLU(),
            BatchNormalization(),
            # Third SimpleRNN layer with ReLU
            SimpleRNN(768, activation="linear"),
            PReLU(),
            BatchNormalization(),
            Dense(len(alpha)),
        ]
    )

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "checkpoints/relu_rnn"
    return (model, checkpoint_dir)


def make_model_dense_rnn():
    embedded_output_dim = len(alpha)
    rnn_units = 256
    num_layers = 10  # Easily adjustable number of layers

    # Use Functional API for complex connectivity
    inputs = Input(shape=(window_size,))

    # Embedding layer
    x = Embedding(input_dim=len(alpha), output_dim=embedded_output_dim)(inputs)
    x = BatchNormalization()(x)

    # Store sequence outputs for concatenation
    sequence_outputs = [x]

    # Create RNN layers with dense connectivity
    for i in range(num_layers):
        # Last layer doesn't need to return sequences
        return_sequences = i < num_layers - 1

        # For layers after the first, concatenate all previous sequence outputs
        if i > 0:
            x = Concatenate(axis=2)(sequence_outputs)

        # Apply RNN layer with PReLU and BatchNorm
        x = SimpleRNN(
            rnn_units, activation="linear", return_sequences=return_sequences
        )(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)

        # Add to sequence outputs if it returns sequences
        if return_sequences:
            sequence_outputs.append(x)

    # Final prediction layer
    outputs = Dense(len(alpha))(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "checkpoints/dense_rnn_for_10"
    return (model, checkpoint_dir)


def make_model_condensed_skip_rnn():
    embedded_output_dim = len(alpha)
    rnn_units = 256
    condensed_dim = 256  # Size of the compressed skip connections
    num_layers = 8

    # Use Functional API
    inputs = Input(shape=(window_size,))

    # Embedding layer
    x = Embedding(input_dim=len(alpha), output_dim=embedded_output_dim)(inputs)
    x = LayerNormalization()(x)

    # Store all sequence outputs for skip connections
    all_outputs = [x]
    current_output = x

    # Create RNN layers with learnable condensed skip connections
    for i in range(num_layers):
        # All layers return sequences now
        return_sequences = True

        # Condense half of previous outputs through a learnable projection
        if i > 0:
            # Concatenate half of previous outputs
            combined = Concatenate(axis=2)(all_outputs[-1::-2])

            # Learnable projection to reduce dimensionality
            skip_projection = TimeDistributed(
                Dense(condensed_dim, activation="linear")
            )(combined)
            skip_projection = LayerNormalization()(skip_projection)
            skip_projection = PReLU()(skip_projection)

            # Feed the condensed representation to the RNN layer
            current_output = SimpleRNN(
                rnn_units,
                activation="linear",
                kernel_initializer=Orthogonal(gain=1.2),
                recurrent_initializer=Orthogonal(gain=1.2),
                return_sequences=return_sequences,
            )(skip_projection)
        else:
            # First layer just processes the embedding
            current_output = SimpleRNN(
                rnn_units,
                activation="linear",
                kernel_initializer=Orthogonal(gain=1.2),
                recurrent_initializer=Orthogonal(gain=1.2),
                return_sequences=return_sequences,
            )(current_output)

        current_output = LayerNormalization()(current_output)
        current_output = PReLU()(current_output)

        # Save this output for future skip connections
        all_outputs.append(current_output)

    # Final prediction layer - predict at each timestep
    outputs = TimeDistributed(Dense(len(alpha)))(Concatenate(axis=2)(all_outputs))

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(global_clipnorm=1.0),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "checkpoints/condensed_skip_rnn_layer_full_sequence_8"
    return (model, checkpoint_dir)


def main():
    setup()

    # Build a simple model.
    model, checkpoint_dir = make_model_gru_skip()
    # dataset = make_data()
    dataset = RandomSubsetSequence()

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir, "my_model_epoch_{epoch:02d}_batch_{batch:05d}.keras"
        ),  # This filename pattern includes the epoch and batch number.
        monitor="loss",  # You can change this to any metric, e.g. 'val_loss'
        verbose=1,
        save_best_only=False,
        save_freq="epoch",  # Save every 1000 samples processed.
    )

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Restoring from", latest_checkpoint)
        model.load_weights(latest_checkpoint)
    else:
        print("No checkpoint found. Training from scratch.")

    tensorboard_cb = TensorBoard(
        log_dir="./beam-logs",  # Directory where the logs will be saved.
        histogram_freq=0,  # Frequency (in epochs) at which to compute activation and weight histograms.
        write_graph=False,  # Save the graph visualization.
        update_freq="epoch",  # Update frequency, can also be an integer (e.g. number of batches).
    )

    # Start training.
    # Note: Depending on the size of your dataset, you might need to adjust steps_per_epoch.
    model.fit(dataset, epochs=10_000, callbacks=[checkpoint_cb, tensorboard_cb])


if __name__ == "__main__":
    main()
