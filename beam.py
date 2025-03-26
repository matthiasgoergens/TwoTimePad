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
    ZeroPadding1D,
    ZeroPadding2D,
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

    # No need to batch again — it's already batched!
    return dataset.prefetch(tf.data.AUTOTUNE)


from tensorflow.keras.layers import Layer
import tensorflow as tf


class PartialResidualAdd(Layer):
    def __init__(self, **kwargs):
        super(PartialResidualAdd, self).__init__(**kwargs)

    def build(self, input_shape):
        # Validate input shape
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("Input must be a list of two tensors")

        # Mark the layer as built
        self.built = True

    def call(self, inputs):
        # inputs is a list [lstm_output, residual]
        lstm_output, residual = inputs

        # Get the number of channels
        lstm_channels = tf.shape(lstm_output)[-1]
        residual_channels = tf.shape(residual)[-1]

        # Hmm, this code is run every time!
        # # Debug print
        # tf.print(
        #     "LSTM channels:", lstm_channels, "Residual channels:", residual_channels
        # )

        # Expected case: lstm_output has more or equal channels than residual
        # if lstm_channels >= residual_channels:
        # Split the lstm_output into matching part and remainder
        matching_part = lstm_output[..., :residual_channels]
        remainder_part = lstm_output[..., residual_channels:]

        # Add the residual to the matching part
        added_part = matching_part + residual

        # Concatenate the added part with the remainder
        result = tf.concat([added_part, remainder_part], axis=-1)
        return result

    def compute_output_shape(self, input_shape):
        # Output shape will be the larger of the two input shapes
        if input_shape[0][-1] >= input_shape[1][-1]:
            return input_shape[0]
        else:
            return input_shape[1]


def make_model_lstm_skip():
    """
    Insights:
    - LayerNormalisation before hitting the LSTM layer, but not in the residual 'path'.
    - BatchNormalisation instead of LayerNormalisation seems to make learning a lot faster,
      at least at first.  Let's see.

    Also try PReLU instead of LSTM.
    Also consider mixing Residual with skip connections?
    Or growing the residual over layers?
    """
    num_layers = 10
    units = 1024

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
        lstm_output = LSTM(units, return_sequences=True, name=f"lstm_{i}")(
            BatchNormalization()(next_input)
        )
        print(f"{i} units: {units}\t{next_input}\t{lstm_output}")
        # Use the adjust_add helper to perform the residual connection.
        # next_input = adjust_add(lstm_output, next_input)

        next_input = PartialResidualAdd()([lstm_output, next_input])

    # Output layer - predict at each timestep
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
    checkpoint_dir = "lstm_residual_never_norm_residual_batchnorm_growing_4"
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
