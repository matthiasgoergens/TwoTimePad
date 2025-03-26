import os
import os.path
import random

import numpy as np

# Monkey patching to make np.inf work with TensorFlow.
np.Inf = np.inf

import subprocess

import tensorflow as tf
import tensorflow.keras.saving as saving
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
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
    ZeroPadding1D,
    ZeroPadding2D,
    average,
    concatenate,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.mixed_precision import set_global_policy
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


class PartialResidualAdd(Layer):
    """
    TODO: consider split, add, concat; instead of padding.
    """

    def __init__(self, **kwargs):
        super(PartialResidualAdd, self).__init__(**kwargs)
        self.add_layer = Add()

    def call(self, inputs):
        # inputs is a list [x, residual] where x has more channels than residual
        x, residual = inputs

        # Get shapes
        x_shape = tf.shape(x)
        res_shape = tf.shape(residual)

        # Create a padded version of residual with zeros in the extra channels
        padding = [[0, 0], [0, 0], [0, x_shape[-1] - res_shape[-1]]]
        padded_residual = tf.pad(residual, padding)

        # Add the padded residual to x
        return self.add_layer([x, padded_residual])


import tensorflow as tf
from tensorflow.keras.layers import Layer


class BlockDropout(Layer):
    """
    Implements stochastic depth by randomly zeroing out the entire tensor
    with probability drop_rate during training.

    This is meant to be used before a residual connection.
    """

    def __init__(self, drop_rate=0.2, **kwargs):
        """
        Args:
            drop_rate: Float between 0 and 1. Probability of zeroing out the input.
        """
        super(BlockDropout, self).__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, inputs, training=None):
        # During inference or if drop_rate is 0, return unchanged
        if not training or self.drop_rate == 0:
            return inputs

        # Create a random binary tensor: 1 with probability (1-drop_rate), 0 with probability drop_rate
        batch_size = tf.shape(inputs)[0]
        random_tensor = tf.random.uniform([batch_size], 0, 1)
        binary_tensor = tf.cast(random_tensor >= self.drop_rate, inputs.dtype)

        # Reshape for broadcasting to all dimensions
        ndims = len(inputs.shape)
        broadcast_shape = [batch_size] + [1] * (ndims - 1)
        binary_tensor = tf.reshape(binary_tensor, broadcast_shape)

        # Scale the kept values to maintain the same expected value
        keep_prob = 1.0 - self.drop_rate
        outputs = inputs * binary_tensor / keep_prob

        return outputs

    def get_config(self):
        config = super(BlockDropout, self).get_config()
        config.update({"drop_rate": self.drop_rate})
        return config


def make_model_lstm_skip():
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
    num_layers = 10
    units = 1.5 * 1024

    layer_units = [
        len(alpha) + round(i * (units - len(alpha)) / num_layers)
        for i in range(1, num_layers + 1)
    ]

    # Input layer
    inputs = Input(shape=(window_size,))

    # Embedding layer
    embed = Embedding(input_dim=len(alpha), output_dim=len(alpha))(inputs)

    # Weirdly, this code is run again and again.
    print("\nLayers!\n")
    # Store layer outputs for skip connections
    next_input = embed
    for i, units in enumerate(layer_units):
        # Create LSTM layer; note that we use BatchNormalization only before the LSTM.
        normed = BatchNormalization()(next_input)

        rnn_layer = GRU(units, return_sequences=True, name=f"rnn_{i}")

        lstm_output = rnn_layer(normed)
        # lstm_output = PReLU()(lstm_output)
        # lstm_output = BlockDropout(drop_rate=1 / num_layers)(lstm_output)
        print(f"{i} units: {units}\t{next_input}\t{lstm_output}")
        # Use the adjust_add helper to perform the residual connection.
        # next_input = adjust_add(lstm_output, next_input)

        next_input = PartialResidualAdd()([lstm_output, next_input])

    # Experiment TODO: add BatchNormalization before or after the dense layer here.
    # Output layer - predict at each timestep
    # outputs = TimeDistributed(Dense(len(alpha)))(BatchNormalization()(next_input))
    outputs = TimeDistributed(Dense(len(alpha)))(next_input)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(
            global_clipnorm=0.5,
            weight_decay=1e-4,
        ),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    checkpoint_dir = "rnn_gru"
    return model, checkpoint_dir


def main():
    setup()

    # Build a simple model.
    if True:
        model, model_name = make_model_lstm_skip()
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

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "my_model_epoch_{epoch:02d}.keras"),
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
        epochs=1_000_000,
        callbacks=[
            checkpoint_cb,
            tensorboard_cb,
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, cooldown=50),
        ],
        # initial_epoch=6,
        steps_per_epoch=10,
    )


if __name__ == "__main__":
    main()

# >>> import math
# >>> math.log(46)/2
# 1.9143206982445475
