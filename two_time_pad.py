# -*- coding: utf-8 -*-

import functools as ft
import itertools as it
import math
import random
import re
import sys
from datetime import datetime
from pprint import pprint

import numpy as np
import tensorflow as tf
# tf.config.threading.set_inter_op_parallelism_threads(0)
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras

from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    LSTM,
    Add,
    Average,
    BatchNormalization,
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
    SeparableConv1D,
    SimpleRNN,
    Softmax,
    SpatialDropout1D,
    TimeDistributed,
    average,
    concatenate,
)
from tensorflow.keras.models import Model, Sequential

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    useGPU = False
    print(SystemError("GPU device not found", device_name))
    raise NotImplementedError("Want GPU")
else:
    useGPU = True
    print("Found GPU at: {}".format(device_name))


# from tensorflow.keras.mixed_precision import experimental as mixed_precision

np.set_printoptions(precision=4)

alpha = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()".lower()
alphaRE = alpha.replace("-", "\\-")

assert len(alpha) == 46

accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def nAccuracy(y_true, y_pred):
    return 1 - accuracy(y_true, y_pred)


def error(y_true, y_pred):
    return 1 - accuracy(y_true, y_pred)


def sumError(y_true, y_pred):
    # raise TabError((y_true, y_pred))
    # shape = (32, 50)
    output = tf.reduce_mean(y_pred, -1)
    return output


def load():
    # text = ' '.join(f.open('r').read() for f in pathlib.Path('data').glob('*.txt')).lower()
    text = open("corpus.txt", "r").read().lower()
    text = re.sub("\s+", " ", text)
    # text = re.sub(f'[^{alphaRE}]', '', text)
    text = re.sub("[^%s]" % alphaRE, "", text)
    return text


def sub(cipher, key):
    return [(a - b) % len(alpha) for a, b in zip(cipher, key)]


def clean(text):
    t = {c: i for i, c in enumerate(alpha)}
    return [t[c] for c in text]


def toChar(numbers):
    return "".join(alpha[i] for i in numbers)


# toChar
# (5, 125, 46)
def toChars(tensor):
    (linesNums, charNum, alphaNum) = tensor.shape
    output = []
    # TODO: use gather https://www.tensorflow.org/api_docs/python/tf/gather?version=stable
    assert alphaNum == len(alpha)
    for lineNum in range(linesNums):
        chars = []
        for cN in range(charNum):
            (_, char) = max(
                [(tensor[lineNum, cN, alphaN], alphaN) for alphaN in range(alphaNum)]
            )
            chars.append(char)
        output.append(toChar(chars))
    return output


batch_size = 512


def round_to(x, n):
    return (x // n) * n

msra = tf.initializers.VarianceScaling(scale=1 / 10, distribution="truncated_normal")

def make1(window, text):
    (size,) = text.shape
    start = random.randrange(window)
    return tf.reshape(
        tf.slice(
            text, [start], [round_to(size - window * batch_size, window * batch_size)]
        ),
        (-1, window),
    )

class TwoTimePadSequence(keras.utils.Sequence):
    def _load(self):
        self.a = tf.random.shuffle(make1(self.window, self.mtext))
        self.aa = tf.reshape(self.a, (-1, batch_size, self.window))

        self.size = self.aa.shape[0]
        self.items = iter(range(self.size))

    def on_epoch_end(self):
        print(f"Epoch {self.epochs} ended.")
        self._load()
        self.epochs += 1
        # raise NotImplementedError("Called on epoch end")

    def __len__(self):
        return self.aa.shape[0]
        # return self.training_size

    def __getitem__(self, idx):
        i = idx 
        # i = next(self.items, None)
        # # Hack, because on_epoch_end doesn't seem to be called.
        # if i is None:
        #     self._load()
        #     return self.__getitem__(idx)
        # else:
        return (self.aa[i, :, :-1], self.aa[i, :, -1])

    def __init__(
        self, window, training_size, mtext, both=True, dev=False, extra_key=False
    ):
        self.mtext = mtext

        self.epochs = 0
        # self.training_size = training_size
        self.window = window
        self._load()


HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.0, 0.5))
HP_HEIGHT = hp.HParam("height", hp.IntInterval(0, 30))
# HP_blocks = hp.HParam("blocks", hp.IntInterval(0, 30))
HP_WINDOW = hp.HParam("window", hp.IntInterval(1, 100))
# HP_resSize = hp.HParam("resSize", hp.IntInterval(46, 8 * 46))
# HP_bottleneck = hp.HParam("bottleneck", hp.IntInterval(0, 1000))
HP_blowup = hp.HParam("blowup", hp.IntInterval(1, 11))
# HP_max_kernel = hp.HParam("max_kernel", hp.IntInterval(3, 1 + 2 * 9))
# HP_deviation_as_loss = hp.HParam("deviation_weight", hp.RealInterval(0.0, 10.0))

METRIC_ACCURACY = "accuracy"

# relu = ft.partial(tf.keras.layers.PReLU, shared_axes=[1])
relu = tf.keras.layers.PReLU
crelu = lambda: tf.nn.crelu


def plus(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return Add()([a, b])


def concat(l):
    l = [item for item in l if item is not None]
    if len(l) == 1:
        return l[0]
    else:
        return concatenate(l)


def avg(l):
    assert isinstance(l, (list,)), type(l)
    l = [item for item in l if item is not None]
    if len(l) == 1:
        return l[0]
    else:
        return average(l)


def cat(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return concatenate([a, b])

def sequential(*layers):
    def helper(last):
        for layer in layers:
            last = layer(last)
        return last

    return helper

def make_model_simple(hparams):
    n = hparams[HP_WINDOW] - 1
    height = hparams[HP_HEIGHT]
    blowup = hparams[HP_blowup]
    # height = 4

    inputA = Input(shape=(n,), name="prefix", dtype="int32")
    base = 10
    embeddedA = Embedding(
        output_dim=base,
        input_length=n,
        input_dim=len(alpha),
        name="embeddingA",
        batch_input_shape=[batch_size, n],
    )(inputA)

    outputs = Flatten()(embeddedA)
    # outputs = Dropout(rate=hparams[HP_DROPOUT])(outputs)
    for i in range(height):
        outputs = cat(
            outputs,
            Sequential(
                [
                    BatchNormalization(),
                    relu(),
                    Dense(blowup),
                    # Dropout(rate=hparams[HP_DROPOUT]),
                ]
            )(outputs),
        )
    make_end = lambda name: Sequential(
        [
            relu(),
            Dropout(rate=hparams[HP_DROPOUT]),
            Dense(len(alpha)),
        ],
        name=name,
    )
    clear = make_end("predict")(outputs)
    model = Model([inputA], [clear])

    model.compile(
        optimizer=tf.optimizers.Adam(),
        # optimizer=tf.optimizers.Adam(learning_rate=0.001),
        # optimizer=tf.optimizers.Adam(),
        # optimizer=tf.keras.optimizers.experimental.Nadam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss_weights={'clear': 1/2, 'key': 1/2},
        metrics=[nAccuracy],
    )
    return model

l = 50
hparams = {
    HP_DROPOUT: 0.5,
    HP_HEIGHT: 50,
    HP_WINDOW: l,
    HP_blowup: 46,
}

weights_name = "2023-deep.h5"

make_model = make_model_simple


def show():
    make_model(hparams).summary()


def showOld():
    keras.models.load_model("weights/" + weights_name).summary()


def main(predict_only=False):
    # TODO: Actually set stuff to float16 only, in inference too.  Should use
    # less memory.
    # policy = mixed_precision.Policy("mixed_float16")
    # policy = mixed_precision.Policy("float32")
    # mixed_precision.set_policy(policy)
    # print("Compute dtype: %s" % policy.compute_dtype)
    # print("Variable dtype: %s" % policy.variable_dtype)
    with tf.device(device_name):
        text = clean(load())
        # mtext = tf.convert_to_tensor(text)
        mtext = tf.convert_to_tensor(text)

        # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = "logs/scalars/{}".format(weights_name)
        tensorboard_callback = TensorBoard(
            log_dir=logdir,
            update_freq=50_000,
            profile_batch=0,
        )

        checkpoint = ModelCheckpoint(
            "weights/" + weights_name, monitor="loss", verbose=1, save_best_only=True
        )

        callbacks_list = [
            checkpoint,
            tensorboard_callback,
            # hp.KerasCallback(logdir, hparams),
            ReduceLROnPlateau(
                monitor="loss",
                mode="min",
                patience=10,
                cooldown=5,
                factor=1 / 2,
                verbose=1,
                min_delta=0.001,
            ),
            # LearningRateScheduler(schedule),
            EarlyStopping(
                monitor="loss", patience=60, verbose=1, restore_best_weights=True
            ),
        ]

        with tf.summary.create_file_writer("logs/scalars").as_default():
            hp.hparams_config(
                hparams=[HP_HEIGHT, HP_WINDOW],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
            )

        # try:
        print("Making model.")
        model = make_model(hparams)
        try:
            # raise NotImplementedError("Not loading weights for testing.")
            print("Trying to load weights.")
            model.load_weights("weights/" + weights_name)
            model.summary()
            print(weights_name)
            print("Loaded weights.")
        except:
            # raise
            model.save("weights/" + weights_name, include_optimizer=False)
            model.summary()
            print(weights_name)
            print("Failed to load weights.")
            pass
            # raise

        if predict_only:
            model.predict
            raise NotImplementedError("Need to implement beam search, and loading of sample text.")
        else:
            try:
                num_data = 2 * 10 ** 4
                model.fit(
                    x=TwoTimePadSequence(
                        l, round_to(num_data, batch_size), mtext,
                    ),
                    # x = x, y = y,
                    # steps_per_epoch=10 ** 4 // 32,
                    # max_queue_size=10 ** 3,
                    # initial_epoch=0,
                    # epochs=epoch+1,
                    # validation_split=0.1,
                    validation_data=TwoTimePadSequence(
                        l, round_to(num_data // 10, batch_size), mtext,
                    ),
                    epochs=100_000,
                    callbacks=callbacks_list,
                    batch_size=batch_size,
                    verbose=1,
                )
            except:
                print("Saving model...")
                model.save(f"weights/last_{weights_name}", include_optimizer=True)
                print("Saved model.")
                raise

    # Idea: we don't need the full 50% dropout regularization, because our input is already random.
    # So try eg keeping 90% of units?  Just enough to punish big gross / small net co-adaptions.

    # But wow, this bigger network (twice as large as before) trains really well without dropout.  And no learning rate reduction, yet.
    # It's plateau-ing about ~2.54 loss at default learning rate after ~20 epoch.  (If I didn't miss a restart.)

    # adense-6-c46.h5/train and fractal-6-relu-avg-base_8-post-staggered3.h5 and denseCNN-20-random-mixed-pre-activation-shorter-seed-23.h5 are best so far.
    # denseCNN-20-random-mixed-pre-activation-shorter-seed-23.h5 best by far.  That's what I'm trying to recreate and improve on.
    # Both-loss at minimum was ~.92 (so single ~0.46) and accuracy was ~86.2%

    # Dropout _after_ all BatchNorm is fine.  Especially drop out just before the end should help.


# Base loss for one side:
# log(46, 2)
# 5.523561956057013

if __name__ == "__main__":
    if useGPU:
        main(predict_only=False)
    # else:
    #     show()
