import os
import os.path
import random

import numpy as np

# Monkey patching to make np.inf work with TensorFlow.
np.Inf = np.inf

import subprocess
import tensorflow as tf
import tensorflow.keras.saving as saving
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Add,
    Average,
    BatchNormalization,
    Bidirectional,
    Concatenate,
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
    LayerNormalization,
    MaxPooling1D,
    PReLU,
    SeparableConv1D,
    SimpleRNN,
    Softmax,
    SpatialDropout1D,
    TimeDistributed,
    average,
    concatenate,
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


@saving.register_keras_serializable()
class LastCharLoss(tf.keras.metrics.Mean):
    def __init__(self, name="last_char_loss", **kwargs):
        super(LastCharLoss, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(
            y_true[-1], y_pred[-1], from_logits=True
        )
        return super(LastCharLoss, self).update_state(loss_value, sample_weight)


# window_size * subset_size ~ 10M
window_size = 100
batch_size = 64
subset_size = 100_000

corpus_filename = "corpus.bytes"

record_size = window_size + 1
dtype = tf.uint8


def batched_generator():
    proc = subprocess.Popen(
        [
            "cargo",
            "run",
            "--release",
            "--",
            "generate-snippets",
            f"{record_size}",
            f"../{corpus_filename}",
        ],
        cwd="convert_corpus",
        stdout=subprocess.PIPE,
        bufsize=record_size * batch_size * 4,  # Optional: increase buffer size
    )

    while True:
        data = proc.stdout.read(record_size * batch_size)
        if not data or len(data) < record_size * batch_size:
            print("Unexpected EOF in dataset generator")
            proc.terminate()
            return

        # Decode raw bytes to tensor
        batch = tf.convert_to_tensor(
            [
                list(data[i * record_size : (i + 1) * record_size])
                for i in range(batch_size)
            ],
            dtype=tf.uint8,
        )

        x = batch[:, :-1]
        y = batch[:, 1:]
        # # debug output:
        # for i in range(3):
        #     xx = x[i].numpy().tolist()
        #     yy = y[i].numpy().tolist()
        #     print(f"x[{i}]:", xx)
        #     print(f"y[{i}]:", yy)
        #     print(''.join(alpha[j] for j in xx))
        #     print(''.join(alpha[j] for j in yy))
        yield x, y  # Directly yield input-target pair


def make_data():
    # Read the file one byte at a time.
    dataset = tf.data.FixedLengthRecordDataset(corpus_filename, record_bytes=1)
    # Dataset signature: batch of shape (batch_size, record_size)
    dataset = tf.data.Dataset.from_generator(
        batched_generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, window_size), dtype=dtype),  # x
            tf.TensorSpec(shape=(batch_size, window_size), dtype=dtype),  # y
        ),
    )

    # No need to batch again — it's already batched!
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_model_gru_skip():
    embedded_output_dim = 46
    units = 256

    # Layer configuration - all return sequences now
    layer_config = [
        {"units": units, "skip_from": [], "skip_type": None},
        {"units": units, "skip_from": [0], "skip_type": "residual"},
        {"units": units, "skip_from": [0, 1], "skip_type": "concat"},
        {"units": units, "skip_from": [0, 1, 2], "skip_type": "concat"},
        {"units": units, "skip_from": [1, 2, 3], "skip_type": "concat"},
        {"units": units, "skip_from": [2, 3, 4], "skip_type": "concat"},
        {"units": units, "skip_from": [3, 4, 5], "skip_type": "concat"},
        {"units": units, "skip_from": [4, 5, 6], "skip_type": "concat"},
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
        optimizer=tf.optimizers.Adam(global_clipnorm=1.0, weight_decay=1e-4),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", LastCharLoss()],
    )

    model.summary()
    checkpoint_dir = (
        "checkpoints/gru_to_final_sequence_weight_decay_larger_window_skip_5"
    )
    return (model, checkpoint_dir)

def main():
    setup()

    # Build a simple model.
    if True:
        model, checkpoint_dir = make_model_gru_skip()
    else:
        checkpoint_dir = (
            "checkpoints/gru_to_final_sequence_weight_decay_larger_window_skip_5"
        )
        path = os.path.join(checkpoint_dir, "my_model_epoch_05.keras")
        model = tf.keras.models.load_model(
            path, custom_objects={"LastCharLoss": LastCharLoss}
        )

        model.summary()
    # dataset = RandomSubsetSequence()
    dataset = make_data()

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "my_model_epoch_{epoch:02d}.keras"),
        monitor="loss",  # You can change this to any metric, e.g. 'val_loss'
        verbose=1,
        save_best_only=False,
        # save_freq="epoch",  # Save every 1000 samples processed.
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
        # update_freq="epoch",  # Update frequency, can also be an integer (e.g. number of batches).
    )

    # Start training.
    # Note: Depending on the size of your dataset, you might need to adjust steps_per_epoch.
    model.fit(
        dataset,
        epochs=10_000,
        callbacks=[checkpoint_cb, tensorboard_cb],
        # initial_epoch=6,
        steps_per_epoch=1_000,
    )


if __name__ == "__main__":
    main()
