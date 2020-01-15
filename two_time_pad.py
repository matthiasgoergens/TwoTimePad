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

# import tensorflow_addons as tfa
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras.layers import (
    LSTM,
    Add,
    Average,
    average,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GaussianDropout,
    Embedding,
    Flatten,
    GlobalMaxPooling1D,
    Input,
    MaxPooling1D,
    SeparableConv1D,
    SimpleRNN,
    Softmax,
    SpatialDropout1D,
    TimeDistributed,
    concatenate,
    Layer,
    Lambda,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow_addons.layers import Maxout
import tensorflow_addons as tfa

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    useGPU = False
    print(SystemError("GPU device not found", device_name))
    # raise NotImplementedError("Want GPU")
else:
    useGPU = True
    print("Found GPU at: {}".format(device_name))


from tensorflow.keras.mixed_precision import experimental as mixed_precision


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


batch_size = 64


def round_to(x, n):
    return (x // n) * n


def make1(window, text):
    (size,) = text.shape
    start = random.randrange(window)
    return tf.reshape(
        tf.slice(
            text, [start], [round_to(size - window * batch_size, window * batch_size)]
        ),
        (-1, window),
    )


def makeEpochs(mtext, window, ratio):
    while True:
        x = make1(window, mtext)
        y = make1(window, mtext)
        (size, _) = x.shape
        training_size = round(size * ratio)
        for _ in range(100):
            xx = tf.random.shuffle(x)
            yy = tf.random.shuffle(y)
            cipherX = (xx - yy) % 46
            cipherY = (yy - xx) % 46
            # Drop last epoch, it's probably not full.
            for i in list(range(0, x.shape[0], training_size))[:-1]:
                yield (
                    cipherX[i : i + training_size, :],
                    cipherY[i : i + training_size, :],
                ), (
                    xx[i : i + training_size, :],
                    yy[i : i + training_size, :],
                )


class TwoTimePadSequence(keras.utils.Sequence):
    def _load(self):
        self.aa = tf.reshape(tf.random.shuffle(self.a), (-1, batch_size, self.window))
        self.bb = tf.reshape(tf.random.shuffle(self.b), (-1, batch_size, self.window))
        self.cipherA = (self.aa - self.bb) % 46
        self.cipherB = (self.bb - self.aa) % 46

        self.size = self.aa.shape[0]
        self.items = iter(range(self.size))

    def on_epoch_end(self):
        print("Epoch {self.epochs} ended.")
        self._load()
        self.epochs += 1
        raise NotImplementedError("Called on epoch end")

    def __len__(self):
        return self.training_size

    def __getitem__(self, idx):
        i = next(self.items, None)
        # Hack, because on_epoch_end doesn't seem to be called.
        if i is None:
            self._load()
            return self.__getitem__(idx)
        else:
            if self.both and not self.dev:
                return (
                    (self.cipherA[i, :, :], self.cipherB[i, :, :]),
                    (self.aa[i, :, :], self.bb[i, :, :]),
                )
            elif self.both and self.dev:
                return (
                    (self.cipherA[i, :, :], self.cipherB[i, :, :]),
                    (
                        self.aa[i, :, :],
                        self.bb[i, :, :],
                        tf.zeros(
                            (batch_size, self.window), dtype=tf.dtypes.float32
                        ),
                    ),
                )
            else:
                # return (self.cipherA[i, :, :], ), (self.aa[i, :, :], self.bb[i, :, :])
                return (self.cipherA[i, :, :],), (self.aa[i, :, :],)

    def __init__(self, window, training_size, mtext, both=True, dev=False):
        self.a = make1(window, mtext)
        self.b = make1(window, mtext)

        self.epochs = 0
        self.training_size = training_size
        self.window = window
        self._load()
        self.both = both
        self.dev = dev


def cipher_for_predict():
    # remove eol
    c1 = clean(open("TwoTimePad/examples/ciphertext-1.txt", "r").read().lower()[:-1])
    # print(c1)
    c2 = clean(open("TwoTimePad/examples/ciphertext-2.txt", "r").read().lower()[:-1])
    # print(c2)
    return tf.convert_to_tensor([sub(c1, c2)])


HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.0, 0.5))
HP_HEIGHT = hp.HParam("height", hp.IntInterval(0, 30))
HP_blocks = hp.HParam("blocks", hp.IntInterval(0, 30))
HP_WINDOW = hp.HParam("window", hp.IntInterval(1, 100))
HP_resSize = hp.HParam("resSize", hp.IntInterval(46, 8 * 46))
HP_bottleneck = hp.HParam("bottleneck", hp.IntInterval(0, 1000))
HP_blowup = hp.HParam("blowup", hp.IntInterval(1, 8))
HP_max_kernel = hp.HParam("max_kernel", hp.IntInterval(3, 1 + 2 * 9))
HP_deviation_as_loss = hp.HParam("deviation_weight", hp.RealInterval(0.0, 10.0))

METRIC_ACCURACY = "accuracy"

relu = ft.partial(tf.keras.layers.PReLU, shared_axes=[1])
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


msra = tf.initializers.VarianceScaling(scale=1/10, distribution="truncated_normal")


def sequential(*layers):
    def helper(last):
        for layer in layers:
            last = layer(last)
        return last

    return helper

def justShift(tensors):
    clear, shifts = tensors
    r = tf.range(46)
    
    r = tf.broadcast_to(r, tf.shape(clear))
    shifts = tf.broadcast_to(tf.expand_dims(shifts, -1), tf.shape(clear))
    indices = (r -  shifts) % 46

    clearShift = tf.gather(clear, indices, batch_dims=2)
    return clearShift

class JustShift(Layer):
    def call(self, tensors):
        return justShift(tensors)


# TODO: I suspect something is still wrong with my shift function.  Test more!
def ShiftLayer(clear, key, shifts):
    clear = JustShift(dtype='float32')([clear, shifts])

    return abs(clear - key)

# Resnet.
def make_model_simple(hparams):
    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]
    ic = lambda: Sequential(
        [BatchNormalization(), SpatialDropout1D(rate=hparams[HP_DROPOUT]),]
    )
    sd = lambda: SpatialDropout1D(rate=hparams[HP_DROPOUT])

    inputA = Input(shape=(n,), name="ciphertextA", dtype="int32")
    # inputB = Input(shape=(n,), name="ciphertextB", dtype='int32')
    base = 4 * 46
    blowup = 3
    embeddedA = Embedding(
        output_dim=base,
        input_length=n,
        input_dim=len(alpha),
        name="embeddingA",
        batch_input_shape=[batch_size, n],
    )(inputA)

    # Idea: Start first res from ic() or conv.
    # Idea: also give input directly, not just embedding?
    conved = Sequential(
        [
            ic(),
            Conv1D(
                filters=blowup * base,
                kernel_size=9,
                padding="same",
                kernel_initializer=msra,
            ),
        ]
    )(embeddedA)

    outputs = embeddedA
    for i in range(height - 1):
        outputs = cat(outputs, conved)
        conved = plus(
            conved,
            Sequential(
                [
                    Maxout(base),
                    ic(),
                    Conv1D(
                        filters=blowup * base,
                        kernel_size=9,
                        padding="same",
                        kernel_initializer=msra,
                    ),
                ]
            )(conved),
        )
    make_end = lambda name: Sequential(
        [
            Maxout(base),
            ic(),
            Conv1D(
                name="output",
                filters=46,
                kernel_size=1,
                padding="same",
                strides=1,
                dtype="float32",
                kernel_initializer=msra,
            ),
        ],
        name=name,
    )
    clear = make_end("clear")(cat(outputs, conved))
    # key = make_end('key')(cat(outputs, conved))
    model = Model([inputA], [clear])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1),
        # optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss_weights={'clear': 1/2, 'key': 1/2},
        metrics=[nAccuracy],
    )
    return model

def make_model_conv_res(hparams):
    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]
    width = hparams[HP_max_kernel]

    ic = lambda: BatchNormalization()

    inputA = Input(shape=(n,), name="ciphertextA", dtype="int32")
    inputB = Input(shape=(n,), name="ciphertextB", dtype="int32")
    base = hparams[HP_resSize]
    blowup = hparams[HP_blowup]
    embedding = Embedding(
        output_dim=46,
        input_length=n,
        input_dim=len(alpha),
        batch_input_shape=[batch_size, n],
    )
    embeddedA = embedding(inputA)
    embeddedB = embedding(inputB)

    def conv(i):
        if i > 0:
            return Sequential(
                [
                    ic(),
                    relu(),
                    # Maxout(base,
                    Conv1D(
                        filters=base*blowup,
                        kernel_size=width,
                        padding="same",
                        kernel_initializer=msra,
                    ),
                ]
            )
        else:
            return Conv1D(
                        filters=base*blowup,
                        kernel_size=width,
                        padding="same",
                        kernel_initializer=msra,
                    )

    convedA = embeddedA
    convedB = embeddedB
    for i in range(height):
        c = conv(i)
        cA, cB = c(cat(convedA, convedB)), c(cat(convedB, convedA))
        if tuple(convedA.shape) == tuple(cA.shape):
            convedA = plus(convedA, cA)
            convedB = plus(convedB, cB)
        else:
            convedA = cA
            convedB = cB

    make_end = Conv1D(
        name="output",
        filters=46,
        kernel_size=1,
        padding="same",
        strides=1,
        dtype="float32",
        kernel_initializer=msra,
    )

    clear = Layer(name="clear", dtype="float32")(make_end(convedA))
    key = Layer(name="key", dtype="float32")(make_end(convedB))

    model = Model([inputA, inputB], [clear, key])

    deviation_weight = hparams[HP_deviation_as_loss]

    dev = ShiftLayer(clear, key, inputA)
    sdev = Layer(name="dev", dtype="float32")(tf.reduce_mean(dev))
    model.add_loss(sdev * deviation_weight)
    model.add_metric(sdev, name="deviation", aggregation='mean')

    model.compile(
        # optimizer=tf.optimizers.Adam(clipvalue=1),
        # optimizer=tf.optimizers.Adam(amsgrad=True),
        optimizer=tf.optimizers.SGD(momentum=0.999, nesterov=True),
        loss={
            "clear": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "key": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        loss_weights={"clear": 1 / 2, "key": 1 / 2},
        metrics=[error],
    )
    return model



#def make_model_conv(hparams):
#    n = hparams[HP_WINDOW]
#    height = hparams[HP_HEIGHT]
#    width = hparams[HP_max_kernel]
#
#    ic = lambda: (BatchNormalization())
#
#    inputA = Input(shape=(n,), name="ciphertextA", dtype="int32")
#    inputB = Input(shape=(n,), name="ciphertextB", dtype="int32")
#    base = hparams[HP_resSize]
#    blowup = hparams[HP_blowup]
#    embedding = Embedding(
#        output_dim=46,
#        input_length=n,
#        input_dim=len(alpha),
#        batch_input_shape=[batch_size, n],
#    )
#    embeddedA = embedding(inputA)
#    embeddedB = embedding(inputB)
#
#    def conv():
#        return Sequential(
#            [
#                Conv1D(
#                    filters=base,
#                    kernel_size=width,
#                    padding="same",
#                    kernel_initializer=msra,
#                ),
#                ic(),
#                relu(),
#            ]
#        )
#
#    convedA = embeddedA
#    convedB = embeddedB
#    for _ in range(height):
#        c = conv()
#        convedA, convedB = c(cat(convedA, convedB)), c(cat(convedB, convedA))
#
#    make_end = Conv1D(
#        name="output",
#        filters=46,
#        kernel_size=1,
#        padding="same",
#        strides=1,
#        dtype="float32",
#        kernel_initializer=msra,
#    )
#
#    clear = Layer(name="clear", dtype="float32")(make_end(convedA))
#    key = Layer(name="key", dtype="float32")(make_end(convedB))
#
#    b = (BatchNormalization())
#    b = lambda x: x
#
#    dev = ShiftLayer(clear, key, inputA)
#    # assert tuple(dev.shape) in [(None, n, 46), (32, n, 46)], dev
#
#    # assert tuple(key.shape) == (None, n, 46), key
#    # assert tuple(dev.shape) == (None, n, 46), dev
#    # assert tuple(key.shape) == (None, n, 46), key
#
#
#    model = Model([inputA, inputB], [clear, key])
#
#    deviation_weight = hparams[HP_deviation_as_loss]
#    sdev = Layer(name="dev", dtype="float32")(tf.reduce_mean(dev)) * deviation_weight
#    model.add_loss(sdev)
#    model.add_metric(sdev, name="deviation", aggregation='mean')
#
#    model.compile(
#        # optimizer=tf.optimizers.Adam(learning_rate=0.001/2),
#        optimizer=tf.optimizers.Adam(),
#        loss={
#            "clear": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#            "key": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#        },
#        loss_weights={"clear": 1 / 2, "key": 1 / 2},
#        metrics=[error],
#    )
#    return model


def make_model_fractal(hparams):
    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]
    width = hparams[HP_max_kernel]

    ic = lambda: (BatchNormalization())

    inputA = Input(shape=(n,), name="ciphertextA", dtype="int32")
    inputB = Input(shape=(n,), name="ciphertextB", dtype="int32")
    base = hparams[HP_resSize]
    blowup = hparams[HP_blowup]
    embedding = Embedding(
        output_dim=46,
        input_length=n,
        input_dim=len(alpha),
        batch_input_shape=[batch_size, n],
    )
    embeddedA = embedding(inputA)
    embeddedB = embedding(inputB)

    def conv():
        # Went from [x | blowup] * base to base to blowup * base
        # So could use cat?
        # Now: geting from base -> blowup * base -> base
        return Sequential(
            [
                # Idea: parallel of different kernel sizes.  Will save on trainable params.
                ic(),
                Maxout(base),
                Conv1D(
                    filters=blowup * base,
                    kernel_size=width,
                    padding="same",
                    kernel_initializer=msra,
                ),
            ]
        )

    def block(n):
        if n <= 0:
            # None means: no weight in average.
            return lambda *args: [None, None]
        else:
            # f 0 = identity (or conv in paper) # Not Implemented
            # f 1 = conv # to be like paper.
            # f (n+1) = (f n . f n) + conv
            inputA = Input(shape=(n, blowup * base))
            inputB = Input(shape=(n, blowup * base))

            c = conv()
            convA = c(cat(inputA, inputB))
            convB = c(cat(inputB, inputA))

            [blockA, blockB] = block(n - 1)(block(n - 1)([inputA, inputB]))

            return Model([inputA, inputB], [avg([blockA, convA]), avg([blockB, convB])])

    c0 = Conv1D(
        filters=blowup * base,
        kernel_size=width,
        padding="same",
        kernel_initializer=msra,
    )
    cA = c0(embeddedA)
    cB = c0(embeddedB)

    convedA, convedB = block(height)([cA, cB])
    make_end = Conv1D(
        name="output",
        filters=46,
        kernel_size=1,
        padding="same",
        strides=1,
        dtype="float32",
        kernel_initializer=msra,
    )

    clear = Layer(name="clear", dtype="float32")(
        make_end(SpatialDropout1D(rate=hparams[HP_DROPOUT])(convedA))
    )
    key = Layer(name="key", dtype="float32")(
        make_end(SpatialDropout1D(rate=hparams[HP_DROPOUT])(convedB))
    )

    un = tf.unstack(key, axis=-2)
    assert n == len(un), un

    embs = tf.unstack(embeddedA, axis=-2)
    shifts = tf.unstack(inputA, axis=-2)
    tf.roll()

    model = Model([inputA, inputB], [clear, key])

    model.compile(
        # optimizer=tf.optimizers.Adam(learning_rate=0.001/2),
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss_weights={"clear": 1 / 2, "key": 1 / 2},
        metrics=[error],
    )
    return model


# Mixture between fractal and dense.
def make_model_fractal_dense(hparams):
    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]

    ic = lambda: sequential(
        (BatchNormalization()),
        SpatialDropout1D(rate=hparams[HP_DROPOUT]),
    )

    input = Input(shape=(n,), name="ciphertextA", dtype="int32")
    base = hparams[HP_resSize]
    blowup = hparams[HP_blowup]
    embedded = Embedding(
        output_dim=46,
        input_length=n,
        input_dim=len(alpha),
        name="embeddingA",
        batch_input_shape=[batch_size, n],
    )(input)

    def conv(extra):
        def helper(inputs):
            input = avg(inputs)
            max_kernel = hparams[HP_max_kernel]
            try:
                conved = Conv1D(
                    filters=extra,
                    kernel_size=3,
                    padding="same",
                    kernel_initializer=msra,
                )((BatchNormalization())(relu()(input)))
            except:
                print(f"Input: {input}")
                raise
            return [cat(input, conved)]

        return helper

    def block(n):
        if n <= 0:
            assert NotImplementedError
            # Identity.  Work out whether we can/should use conv instead?
            # If we use conv, we can have embedding go to 46 instead of base, I think.
            return avg
        elif n <= 1:
            return conv(base)
        else:
            # f 0 = identity (or conv in paper)
            # f (n+1) = (f n . f n) + conv
            def helper(inputs):
                (_batch_size, _time, input_features) = inputs[-1].shape
                inter_out = block(n - 1)(inputs)
                assert isinstance(inter_out, (list,)), type(inter_out)
                outputs = block(n - 1)(inter_out)
                assert isinstance(outputs, (list,)), type(outputs)
                (_batch_sizeO, _timeO, output_features) = outputs[-1].shape
                assert (_batch_size, _time) == (_batch_sizeO, _timeO), (
                    (_batch_size, _time),
                    (_batch_sizeO, _timeO),
                )
                assert input_features <= output_features, (
                    input_features,
                    output_features,
                )
                try:
                    c = conv(output_features - input_features)(inputs)
                except:
                    print("input, output, diff")
                    print(inputs[-1].shape)
                    print(outputs[-1].shape)
                    print((input_features, output_features))
                    raise
                o = [*c, *outputs]
                assert isinstance(o, (list,)), o
                return o

            return helper

    # Idea: Start first res from ic() or conv.
    # Idea: also give input directly, not just embedding?

    conved = avg(block(height)([embedded]))
    clear = Conv1D(
        name="clear",
        filters=46,
        kernel_size=1,
        padding="same",
        strides=1,
        dtype="float32",
        kernel_initializer=msra,
    )(conved)
    model = Model([input], [clear])

    model.compile(
        # optimizer=tf.optimizers.Adam(learning_rate=0.001/2),
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[nAccuracy],
    )
    return model


def make_model_dense(hparams):
    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]

    ic = lambda: sequential(
        (BatchNormalization()),
        SpatialDropout1D(rate=hparams[HP_DROPOUT] / 2),
        Dropout(rate=hparams[HP_DROPOUT] / 2),
    )

    inputA = Input(shape=(n,), name="ciphertextA", dtype="int32")
    inputB = Input(shape=(n,), name="ciphertextB", dtype="int32")
    base = hparams[HP_resSize]
    blowup = hparams[HP_blowup]
    embedding = Embedding(
        output_dim=46,
        input_length=n,
        input_dim=len(alpha),
        name="embeddingA",
        batch_input_shape=[batch_size, n],
    )
    embeddedA = embedding(inputA)
    embeddedB = embedding(inputB)

    def conv():
        def helper(input):
            conved = Conv1D(
                filters=base, kernel_size=3, padding="same", kernel_initializer=msra
            )(input)
            return ic()(relu()(conved))

        return helper

    def dense1(inputA, inputB):
        # TODO: finish
        return cat(input, conv()(input))

    def denseN(height, input):
        return sequential(*(height * [dense1]))(input)

    # Idea: Start first res from ic() or conv.
    # Idea: also give input directly, not just embedding?

    bottleneck = hparams[HP_bottleneck]
    blocks = hparams[HP_blocks]

    def block(n, input):
        if n <= 0:
            return input
        else:
            output = denseN(height, input)
            # Residual connection for all but first block:
            if 1 < n:
                print(f"Bottlenecking at block {n}.")
                output = Conv1D(
                    filters=bottleneck,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=msra,
                )(output)
            else:
                print(f"No bottlenecking at block {n}.")
            if 1 < n < blocks:
                assert tuple(input.shape) == tuple(output.shape), (
                    input.shape,
                    output.shape,
                )
                print(f"Residual connection at block {n}.")
                output = plus(input, output)
            else:
                print(f"No residual connection at block {n}.")
            return block(n - 1, output)

    conved = block(blocks, embedded)

    make_end = lambda name: sequential(
        Conv1D(
            name=name,
            filters=46,
            kernel_size=1,
            padding="same",
            strides=1,
            dtype="float32",
            kernel_initializer=msra,
        ),
    )
    clear = make_end("clear")(conved)
    model = Model([input], [clear])

    model.compile(
        # optimizer=tf.optimizers.Adam(learning_rate=0.001),
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[nAccuracy],
    )
    return model


def make_model_recreate(hparams):

    relu = ft.partial(tf.keras.layers.PReLU, shared_axes=[1])

    n = hparams[HP_WINDOW]
    inputA = Input(shape=(n,), name="ciphertextA", dtype="int32")
    # inputB = (- inputA) % 46
    inputB = Input(shape=(n,), name="ciphertextB", dtype="int32")
    resSize = hparams[HP_resSize]
    width = hparams[HP_max_kernel]
    height = hparams[HP_HEIGHT]

    embedding = Embedding(
        output_dim=resSize,
        input_length=n,
        input_dim=len(alpha),
        name="my_embedding",
        batch_input_shape=[batch_size, n],
    )

    embeddedA = embedding(inputA)
    embeddedB = embedding(inputB)

    def makeResNetNew(i, channels, _, size):
        fanInput = Input(shape=(n, 4 * size,))
        fan = concatenate(
            [
                Conv1D(
                    filters=round(size / 4),
                    kernel_size=width,
                    padding="same",
                    kernel_initializer=msra,
                )(fanInput)
                for width in [3, 5, 7, 9]
            ]
        )
        m = Model([fanInput], [fan])

        return Sequential(
            [
                # Input(shape=(n,channels,)),
                # SpatialDropout1D(rate=hparams[HP_DROPOUT]), # Not sure whether that's good.
                # TODO: if BatchNormalization is independent for each dimension, we can do post BatchNorm, instead of pre?
                # TODO: try different dropout scheme here, that messes less with variance?
                ## Note: dropout done outside.
                # SpatialDropout1D(rate=hparams[HP_DROPOUT] * i / height),
                # (BatchNormalization()),
                # relu(),
                Conv1D(
                    filters=16 * size,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=msra,
                ),
                # TODO: Might want to drop this intermediate batch norm?  So that dropout doesn't have too much impact on variance.
                TimeDistributed(BatchNormalization()),
                Maxout(4 * size),
                m,
                TimeDistributed(BatchNormalization()),
                Maxout(size),
            ],
            name="resnet{}".format(i),
        )

    def makeResNet(i, channels, width, size):
        return Sequential(
            [
                Input(name=f"res_inputMe_i", shape=(n, channels,)),
                # SpatialDropout1D(rate=hparams[HP_DROPOUT]), # Not sure whether that's good.
                TimeDistributed(BatchNormalization(name='bn1'), name='td1'),
                relu(),
                Conv1D(
                    filters=4 * size,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=msra,
                ),
                TimeDistributed(BatchNormalization(name='bn2'), name='td2'),
                relu(),
                Conv1D(
                    filters=size,
                    kernel_size=width,
                    padding="same",
                    kernel_initializer=msra,
                ),
            ],
            name="resnet{}".format(i),
        )

    random.seed(23)

    def make_block(convedA, convedB):
        for i in range(height):
            catA = concatenate([convedA, convedB])
            catB = concatenate([convedB, convedA])
            (_, _, num_channels) = catA.shape
            (_, _, num_channelsB) = catB.shape
            assert tuple(catA.shape) == tuple(catB.shape), (catA.shape, catB.shape)

            width = 1 + 2 * random.randrange(5, 8)
            size = random.randrange(23, 2 * 46)
            # size = resSize
            resNet = makeResNet(i, num_channels, width, size)
            # resNet = tf.recompute_grad(resNet)

            resA = resNet(catA)
            resB = resNet(catB)

            convedA = cat(convedA, resA)
            convedB = cat(convedB, resB)


        return convedA, convedB

    convedA, convedB = make_block(embeddedA, embeddedB)
    # assert tuple(convedA.shape) == tuple(convedB.shape), (convedA.shape, convedB.shape)

    # TODO: check whether final BatchNorm would help?  (Pre dropout, of course.)
    # TODO: Similar for relu?
    # TODO: Try final Dropout with my other approaches, too.
    # TODO: Try different amounts of final dropout.  Can even try very high amounts, because we have so many dimensions at the end.
    # Approx 1,246 dimensions at the end for something close to `faithful` repro.
    # So could try even 90% dropout.
    make_end = Conv1D(
        filters=46,
        kernel_size=1,
        padding="same",
        strides=1,
        dtype="float32",
        kernel_initializer=msra,
    )
    pre_clear = make_end(convedA)
    pre_key = make_end(convedB)

    clear = Layer(name="clear", dtype="float32")(
            pre_clear)
        # 0.9 * pre_clear + 0.1 *
        #     JustShift(dtype='float32')([
        #         pre_key,
        #         inputB,
        #         ]))

    key = Layer(name="key", dtype="float32")(
            pre_key)
        # 0.9 * pre_key + 0.1 *
        #     JustShift(dtype='float32')([
        #         pre_clear,
        #         inputA,
        #         ]))

   

    model = Model([inputA, inputB], [clear, key])

    # dev = ShiftLayer(pre_clear, pre_key, inputA)
    # sdev = Layer(name="dev", dtype="float32")(tf.reduce_mean(dev))
    # model.add_metric(sdev, name="deviation", aggregation='mean')


    model.compile(
        # optimizer=tf.optimizers.Adam(amsgrad=True),
        optimizer=tf.optimizers.SGD(momentum=0.9, nesterov=True),
        # optimizer=tf.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0), # momentum=0.9, nesterov=True),
        # optimizer=tfa.optimizers.AdamW(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss_weights={"clear": 1/2, "key": 1/2},
        metrics=[error],
    )
    return model


l = 50
hparams = {
    HP_DROPOUT: 0.0,
    HP_HEIGHT: 1,
    HP_blocks: 1,
    HP_bottleneck: 46 * 5,
    ## Idea: skip the first few short columns in the fractal.
    # HP_SKIP_HEIGH: 3,
    HP_WINDOW: l,
    HP_resSize: round_to(10*46, 4),
    HP_blowup: 1,
    HP_max_kernel: 5,
    HP_deviation_as_loss: 0.0,
}

weights_name = "recreate 90-to-10 sgd momentum 0.9 warmup batch64 - recompute7 checkpoint.tf"

make_model = make_model_recreate


def show():
    make_model(hparams).summary()


def showOld():
    keras.models.load_model("weights/" + weights_name).summary()


def main():
    # TODO: Actually set stuff to float16 only, in inference too.  Should use
    # less memory.
    policy = mixed_precision.Policy("mixed_float16")
    # policy = mixed_precision.Policy("float32")
    mixed_precision.set_policy(policy)
    print("Compute dtype: %s" % policy.compute_dtype)
    print("Variable dtype: %s" % policy.variable_dtype)

    with tf.device(device_name):
        text = clean(load())
        # mtext = tf.convert_to_tensor(text)
        mtext = tf.convert_to_tensor(text)

        # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = "logs/scalars/{}".format(weights_name)
        tensorboard_callback = TensorBoard(
            log_dir=logdir, update_freq=50_000, profile_batch=0,
        )

        checkpoint = ModelCheckpoint(
            "weights/" + weights_name, monitor="loss", verbose=1, save_best_only=True
        )

        def schedule(epoch):
            # TODO: Aso try SGD with momentum.
            # default = 0.001 # Adam
            default = 0.01 # SGD


            factor = 2**(epoch-7)
            lr = factor * default

            print(
                f"{weights_name}: Scheduled learning rate for epoch {epoch}: {default} * {lr/default}"
            )
            return lr

        def scheduleRampSGD(epoch):
            # TODO: Aso try SGD with momentum.
            default = 0.005
            # lr = default * (epoch / 2) **2
            lr = default * (epoch + 1)
            # NOTE: 32 was still fine, 64 broke.
            print(
                f"{weights_name}: Scheduled learning rate for epoch {epoch}: {default} * {lr/default}"
            )
            return lr

        def slow(epoch):
            return 0.001
            return 0.001 / 100

        callbacks_list = [
            checkpoint,
            tensorboard_callback,
            # hp.KerasCallback(logdir, hparams),
            # ReduceLROnPlateau(
            #     monitor="loss",
            #     mode="min",
            #     patience=20,
            #     cooldown=10,
            #     factor=1 / 2,
            #     verbose=1,
            #     min_delta=0.001,
            # ),
            LearningRateScheduler(schedule),
            EarlyStopping(
                monitor="loss", patience=60, verbose=1, restore_best_weights=True
            ),
        ]

        with tf.summary.create_file_writer("logs/scalars").as_default():
            hp.hparams_config(
                hparams=[HP_DROPOUT, HP_HEIGHT, HP_WINDOW],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
            )

        # try:
        print("Making model.")
        model = make_model(hparams)
        try:
            raise NotImplementedError("Not loading weights for testing.")
            print("Trying to load weights.")
            model.load_weights("weights/" + weights_name)
            model.summary()
            print(weights_name)
            print("Loaded weights.")
        except:
            model.summary()
            model.save("weights/" + weights_name, include_optimizer=False)
            print(weights_name)
            print("Failed to load weights.")
            pass
            # raise
        # model = keras.models.load_model('weights/'+weights_name)

        # for i in range(10*(layers+1)):

        # print("Predict:")
        # predict_size = 10

        # ita_cipher = cipher_for_predict()
        # [ita_label, ita_key] = model .predict(ita_cipher)
        # print(toChars_labels(ita_cipher))
        # pprint(toChars(ita_label)[:1000])
        # pprint(toChars(ita_key)[:1000])

        # (ciphers_p, labels_p, keys_p) = samples(text, predict_size, l)
        # [pred_label, pred_key] = model.predict(ciphers_p)
        # # clear, key, prediction
        # pprint(
        #     list(
        #         zip(
        #             toChars_labels(ciphers_p),
        #             toChars_labels(labels_p),
        #             toChars(pred_label),
        #             toChars_labels(keys_p),
        #             toChars(pred_key),
        #             predict_size * [l * " "],
        #         )
        #     ),
        #     width=120,
        # )

        # model.evaluate(TwoTimePadSequence(l, 10**4 // 32), callbacks=[tensorboard_callback])
        # print("Training:")
        # (ciphers, labels, keys) = samples(text, training_size, l)
        # print(model.fit(ciphers, [labels, keys],
        #        for epoch, (x, y) in enumerate(makeEpochs(mtext, l, 1/60)):
        #           print(f"My epoch: {epoch}")
        if True:
            try:
                model.fit(
                    x=TwoTimePadSequence(l, 10 ** 2, mtext, both=True, dev=False),
                    # x = x, y = y,
                    # steps_per_epoch=10 ** 4 // 32,
                    max_queue_size=10 ** 3,
                    initial_epoch=0,
                    # epochs=epoch+1,
                    # validation_split=0.1,
                    # validation_data=TwoTimePadSequence(
                    #     l, 10 ** 3 // 32, mtext, both=True, dev=False
                    # ),
                    epochs=100_000,
                    callbacks=callbacks_list,
                    # batch_size=batch_size,
                    verbose=1,
                )
            except:
                print("Saving model...")
                model.save("weights/" + weights_name, include_optimizer=False)
                print("Saved model.")
                raise

    # Idea: we don't need the full 50% dropout regularization, because our input is already random.
    # So try eg keeping 90% of units?  Just enough to punish big gross / small nettto co-adaptions.

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
        main()
    else:
        show()
