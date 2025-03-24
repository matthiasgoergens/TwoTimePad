# First: prepare the model.  And prepare some data.
from pprint import pp

import os.path
import numpy as np

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


# Define the window size.
# We want 10 input bytes and 1 target byte.
window_size = 30
total_window_size = window_size + 1
batch_size = 128

corpus_filename = "corpus.bytes"


def make_data():
    # Read the file one byte at a time.
    # FixedLengthRecordDataset will yield records of length 1.
    dataset = tf.data.FixedLengthRecordDataset(corpus_filename, record_bytes=1)

    # Decode the raw bytes into uint8 values.
    # Each element in the dataset will be a scalar representing a byte.
    dataset = dataset.map(lambda x: tf.io.decode_raw(x, tf.uint8)[0])

    # Create sliding windows of total_window_size with a stride of 1.
    # drop_remainder=True ensures every window has exactly total_window_size elements.
    windowed_dataset = dataset.window(total_window_size, shift=1, drop_remainder=True)
    windowed_dataset = windowed_dataset.flat_map(
        lambda window: window.batch(total_window_size)
    )

    # Split each window into (input, target) where input is the first 10 bytes and target is the 11th.
    def split_input_target(window):
        input_bytes = window[:-1]
        target_byte = window[-1]
        return input_bytes, target_byte

    dataset = windowed_dataset.map(split_input_target)

    # (Optional) Shuffle and batch the dataset.
    # Adjust the shuffle buffer size and batch size according to your hardware.
    dataset = (
        dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
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
    embedded_output_dim = 64

    model = Sequential(
        [
            Input(shape=(window_size,)),
            Embedding(input_dim=len(alpha), output_dim=embedded_output_dim),
            BatchNormalization(),
            GRU(512, return_sequences=True),
            BatchNormalization(),
            GRU(512),
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
    checkpoint_dir = "checkpoints/gru_bn3_twice"
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
        return_sequences = i < num_layers - 1

        # Condense half of previous outputs through a learnable projection
        if i > 0:
            # Concatenate half of previous outputs
            combined = Concatenate(axis=2)(all_outputs[-1::-2])

            # Learnable projection to reduce dimensionality
            skip_projection = Dense(condensed_dim, activation="linear")(combined)
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
        if return_sequences:
            all_outputs.append(current_output)

    # Final prediction layer
    outputs = Dense(len(alpha))(current_output)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "checkpoints/condensed_skip_rnn_layer_orthogonal12_6layers_sparse"
    return (model, checkpoint_dir)


def main():
    setup()
    dataset = make_data()

    # Build a simple model.
    model, checkpoint_dir = make_model_condensed_skip_rnn()

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir, "my_model_epoch_{epoch:02d}_batch_{batch:05d}.keras"
        ),  # This filename pattern includes the epoch and batch number.
        monitor="loss",  # You can change this to any metric, e.g. 'val_loss'
        verbose=1,
        save_best_only=False,
        save_freq=10_000,  # Save every 1000 samples processed.
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
        update_freq=1_000,  # Update frequency, can also be an integer (e.g. number of batches).
    )

    # Start training.
    # Note: Depending on the size of your dataset, you might need to adjust steps_per_epoch.
    model.fit(dataset, epochs=10, callbacks=[checkpoint_cb, tensorboard_cb])


if __name__ == "__main__":
    main()
