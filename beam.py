import os
import os.path
import random

import numpy as np

# Monkey patching to make np.inf work with TensorFlow.
np.Inf = np.inf

import subprocess

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import (
    Dropout,
    LSTM,
    BatchNormalization,
    Dense,
    Embedding,
    Input,
    TimeDistributed,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.models import Model, Sequential

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

        set_global_policy("mixed_float16")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


window_size = 150
batch_size = 64

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
        yield x, y


def make_data():
    # Dataset signature: batch of shape (batch_size, record_size)
    dataset = tf.data.Dataset.from_generator(
        batched_generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, window_size), dtype=dtype),  # x
            tf.TensorSpec(shape=(batch_size, window_size), dtype=dtype),  # y
        ),
    )

    # Add sample weights here
    def add_sample_weight(x, y):
        ignore_first = 10
        sample_weight = tf.concat(
            [
                tf.zeros((batch_size, ignore_first)),  # ignore first few timesteps
                tf.ones((batch_size, window_size - ignore_first)),
            ],
            axis=-1,
        )
        return x, y, sample_weight

    dataset = dataset.map(add_sample_weight, num_parallel_calls=tf.data.AUTOTUNE)

    # No need to batch again — it's already batched!
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_model():
    """
    Insights:
    - LayerNormalisation before hitting the LSTM layer, but not in the residual 'path'.
    - BatchNormalisation instead of LayerNormalisation seems to make learning a lot faster,
      at least at first.  Let's see.
    - Ignoring the first 10 (out of 100) losses seems to help with speed of learning.
      It lets the LSTM build up context.
    - Adding a BatchNorm before the final dense projection to len(alpha) seems to not make training progress slower.
      I suspect that's because the benefits of BN are outweighed by making the residual connection with the output worse.

    Also try PReLU instead of LSTM.
    Also consider mixing Residual with skip connections?
    Or growing the residual over layers?
    """
    units = 2048

    model = Sequential(
        [
            Input(shape=(window_size,)),
            Embedding(input_dim=len(alpha), output_dim=len(alpha)),
            BatchNormalization(),
            LSTM(units, return_sequences=True),
            BatchNormalization(),
            LSTM(units, return_sequences=True),
            Dropout(0.05),
            TimeDistributed(Dense(len(alpha))),
        ],
    )

    model.compile(
        optimizer=tf.optimizers.Adam(
            global_clipnorm=0.5,
            weight_decay=1e-5,
        ),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    checkpoint_dir = "lstm_ablated_double_layers_0p05dropout_long"
    return model, checkpoint_dir


def main():
    setup()

    # Build a simple model.
    if True:
        model, model_name = make_model()
        checkpoint_dir = f"checkpoints/{model_name}"
        import random

        log_dir = f"logs/{model_name}/{random.randrange(10_000)}"
    else:
        return
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
    val_data = make_data()

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "epoch_{epoch:02d}.keras"),
        monitor="loss",  # You can change this to any metric, e.g. 'val_loss'
        verbose=1,
        save_best_only=True,
        # save_freq="epoch",  # Save every 1000 samples processed.
    )

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print("Restoring from", latest_checkpoint)
        model.load_weights(latest_checkpoint)
    else:
        print("No checkpoint found. Training from scratch.")

    # import datetime
    # log_dir = f"beam-logs/train/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # tensorboard_cb = TensorBoard(log_dir=log_dir)
    tensorboard_cb = TensorBoard(
        log_dir=log_dir,
        # log_dir="./beam-logs",  # Directory where the logs will be saved.
        # histogram_freq=0,  # Frequency (in epochs) at which to compute activation and weight histograms.
        # write_graph=False,  # Save the graph visualization.
        update_freq="epoch",
        write_steps_per_second=True,
    )

    # Start training.
    # Note: Depending on the size of your dataset, you might need to adjust steps_per_epoch.
    model.fit(
        dataset,
        validation_data=val_data,
        epochs=1_000_000,
        callbacks=[
            checkpoint_cb,
            tensorboard_cb,
            ReduceLROnPlateau(
                monitor="loss", factor=0.5**0.5, patience=50, cooldown=50
            ),
        ],
        steps_per_epoch=80,
        validation_steps=10,
        validation_batch_size=16,
    )


if __name__ == "__main__":
    main()

# >>> import math
# >>> math.log(46)/2
# 1.9143206982445475
