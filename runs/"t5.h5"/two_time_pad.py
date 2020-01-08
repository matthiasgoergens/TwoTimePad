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
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras.layers import (
    LSTM,
    Add, Average, average,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
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
)
from tensorflow.keras.models import Model, Sequential
from tensorflow_addons.layers import Maxout
import tensorflow_addons as tfa

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    useGPU = False
    print(SystemError("GPU device not found", device_name))
    raise NotImplementedError("Want GPU")
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
            (_, char) = max([(tensor[lineNum, cN, alphaN], alphaN) for alphaN in range(alphaNum)])
            chars.append(char)
        output.append(toChar(chars))
    return output

batch_size = 32


def round_to(x, n):
    return (x // n) * n


def make1(window, text):
    (size,) = text.shape
    start = random.randrange(window)
    return tf.reshape(tf.slice(text, [start], [round_to(size - window * batch_size, window * batch_size)]), (-1, window),)




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
                yield (cipherX[i : i + training_size, :], cipherY[i : i + training_size, :]), (
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
            if self.both:
                return (self.cipherA[i, :, :], self.cipherB[i, :, :]), (self.aa[i, :, :], self.bb[i, :, :])
            else:
            # return (self.cipherA[i, :, :], ), (self.aa[i, :, :], self.bb[i, :, :])
                return (self.cipherA[i, :, :], ), (self.aa[i, :, :],)

    def __init__(self, window, training_size, mtext, both=True):
        self.a = make1(window, mtext)
        self.b = make1(window, mtext)

        self.epochs = 0
        self.training_size = training_size
        self.window = window
        self._load()
        self.both = both


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
HP_max_kernel = hp.HParam("max_kernel", hp.IntInterval(3, 1+2*9))

METRIC_ACCURACY = "accuracy"

relu = ft.partial(tf.keras.layers.PReLU, shared_axes=[1])
crelu = lambda: tf.nn.crelu

class CRelu(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(CRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CRelu, self).build(input_shape)

    def call(self, x):
        x = tf.nn.crelu(x, axis=self.axis)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] * 2
        output_shape = tuple(output_shape)
        return output_shape

    def get_config(self, input_shape):
        config = {'axis': self.axis, }
        base_config = super(CReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

msra = tf.initializers.VarianceScaling(scale=2.0, distribution='truncated_normal')

def sequential(*layers):
    def helper(last):
        for layer in layers:
            last = layer(last)
        return last
    return helper

# Resnet.
def make_model_simple(hparams):
    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]
    ic = lambda: Sequential([
        BatchNormalization(),
        SpatialDropout1D(rate=hparams[HP_DROPOUT]),
    ])
    sd = lambda: SpatialDropout1D(rate=hparams[HP_DROPOUT])

    inputA = Input(shape=(n,), name="ciphertextA", dtype='int32')
    # inputB = Input(shape=(n,), name="ciphertextB", dtype='int32')
    base = 4 * 46
    blowup = 3
    embeddedA = Embedding(
        output_dim=base, input_length=n, input_dim=len(alpha), name="embeddingA", batch_input_shape=[batch_size, n],)(
            inputA)

    # Idea: Start first res from ic() or conv.
    # Idea: also give input directly, not just embedding?
    conved = Sequential([
        ic(),
        Conv1D(filters=blowup * base, kernel_size=9, padding='same', kernel_initializer=msra),
    ])(embeddedA)

    outputs = embeddedA
    for i in range(height - 1):
        outputs = cat(outputs, conved)
        conved = plus(conved, Sequential([
            Maxout(base),
            ic(),
            Conv1D(filters=blowup * base, kernel_size=9, padding='same', kernel_initializer=msra),
            ])(conved))
    make_end = lambda name: Sequential([
        Maxout(base),
        ic(),
        Conv1D(name="output", filters=46, kernel_size=1, padding="same", strides=1, dtype='float32', kernel_initializer=msra),
    ], name=name)
    clear = make_end('clear')(cat(outputs, conved))
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


# Mixture between fractal and dense.
def make_model_fractal_dense(hparams):
    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]

    ic = lambda: sequential(
        TimeDistributed(BatchNormalization()),
        SpatialDropout1D(rate=hparams[HP_DROPOUT]),
    )

    input = Input(shape=(n,), name="ciphertextA", dtype='int32')
    base = hparams[HP_resSize]
    blowup = hparams[HP_blowup]
    embedded = Embedding(
        output_dim=46, input_length=n, input_dim=len(alpha), name="embeddingA", batch_input_shape=[batch_size, n],)(
            input)

    def conv(extra):
        def helper(inputs):
            input = avg(inputs)
            max_kernel = hparams[HP_max_kernel]
            try:
                conved = Conv1D(filters=extra, kernel_size=3, padding='same', kernel_initializer=msra)(
                        TimeDistributed(BatchNormalization())(relu()(input)))
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
                inter_out = block(n-1)(inputs)
                assert isinstance(inter_out, (list,)), type(inter_out)
                outputs = block(n-1)(inter_out)
                assert isinstance(outputs, (list,)), type(outputs)
                (_batch_sizeO, _timeO, output_features) = outputs[-1].shape
                assert (_batch_size, _time) == (_batch_sizeO, _timeO), ((_batch_size, _time), (_batch_sizeO, _timeO))
                assert input_features <= output_features, (input_features, output_features)
                try:
                    c = conv(output_features - input_features)(inputs)
                except:
                    print ("input, output, diff")
                    print (inputs[-1].shape)
                    print (outputs[-1].shape)
                    print ((input_features, output_features))
                    raise
                o = [*c, *outputs]
                assert isinstance(o, (list,)), o
                return o
            return helper


    # Idea: Start first res from ic() or conv.
    # Idea: also give input directly, not just embedding?

    conved = avg(block(height)([embedded]))
    clear = Conv1D(name='clear', filters=46, kernel_size=1, padding="same", strides=1, dtype='float32', kernel_initializer=msra)(conved)
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
        TimeDistributed(BatchNormalization()),
        SpatialDropout1D(rate=hparams[HP_DROPOUT]/2),
        Dropout(rate=hparams[HP_DROPOUT]/2),
    )

    inputA = Input(shape=(n,), name="ciphertextA", dtype='int32')
    inputB = Input(shape=(n,), name="ciphertextB", dtype='int32')
    base = hparams[HP_resSize]
    blowup = hparams[HP_blowup]
    embedding = Embedding(
        output_dim=46, input_length=n, input_dim=len(alpha), name="embeddingA", batch_input_shape=[batch_size, n],)
    embeddedA = embedding(inputA)
    embeddedB = embedding(inputB)

    def conv():
        def helper(input):
            conved = Conv1D(filters=base, kernel_size=3, padding='same', kernel_initializer=msra)(input)
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
                output = Conv1D(filters=bottleneck, kernel_size=1, padding='same', kernel_initializer=msra)(output)
            else:
                print(f"No bottlenecking at block {n}.")
            if 1 < n < blocks:
                assert tuple(input.shape) == tuple(output.shape), (input.shape, output.shape)
                print(f"Residual connection at block {n}.")
                output = plus(input, output)
            else:
                print(f"No residual connection at block {n}.")
            return block(n-1, output)

    conved = block(blocks, embedded)


    make_end = lambda name: sequential(
        Conv1D(name=name, filters=46, kernel_size=1, padding="same", strides=1, dtype='float32', kernel_initializer=msra),
    )
    clear = make_end('clear')(conved)
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
    inputA = Input(shape=(n,), name="ciphertextA", dtype='int32')
    inputB = Input(shape=(n,), name="ciphertextB", dtype='int32')
    resSize = hparams[HP_resSize]
    width = hparams[HP_max_kernel]
    height = hparams[HP_HEIGHT]

    embedding = Embedding(output_dim=len(alpha), input_length=n, input_dim=len(alpha), name="my_embedding", batch_input_shape=[batch_size, n],)

    embeddedA = embedding(inputA)
    embeddedB = embedding(inputB)

    def makeResNetNew(i, channels, _, size):
        fanInput = Input(shape=(n,4*size,))
        fan = concatenate([Conv1D(filters=round(size/3), kernel_size=width, padding='same', kernel_initializer=msra)(fanInput) for width in [11, 13, 15]])
        m = Model([fanInput], [fan])

        return Sequential([
            Input(shape=(n,channels,)),

            # SpatialDropout1D(rate=hparams[HP_DROPOUT]), # Not sure whether that's good.
            # TODO: if BatchNormalization is independent for each dimension, we can do post BatchNorm, instead of pre?
            # TODO: try different dropout scheme here, that messes less with variance?

            ## Note: dropout done outside.
            # SpatialDropout1D(rate=hparams[HP_DROPOUT] * i / height),
            relu(),
            Conv1D(filters=4*size, kernel_size=1, padding='same', kernel_initializer=msra),

            # TODO: Might want to drop this intermediate batch norm?  So that dropout doesn't have too much impact on variance.
            TimeDistributed(BatchNormalization()),
            relu(),
            m,
            TimeDistributed(BatchNormalization()),
            ], name="resnet{}".format(i))

    def makeResNet(i, channels, width, size):
        return Sequential([
            Input(name="res_inputMe", shape=(n,channels,)),

            # SpatialDropout1D(rate=hparams[HP_DROPOUT]), # Not sure whether that's good.
            TimeDistributed(BatchNormalization()),
            relu(),
            Conv1D(filters=4*size, kernel_size=1, padding='same', kernel_initializer=msra),

            TimeDistributed(BatchNormalization()),
            relu(),
            Conv1D(filters=size, kernel_size=width, padding='same', kernel_initializer=msra),
            ], name="resnet{}".format(i))


    def make_drop(layers):
        drop = hparams[HP_DROPOUT]
        return list(reversed([
            SpatialDropout1D(drop * distance / height)(layer)
            for distance, layer in enumerate(reversed(layers))]))

    random.seed(23)

    def make_block(convedA, convedB):
        convedAx = [convedA]
        convedBx = [convedB]
        for i in range(height):
            # We deliberately use different dropout masks in all four cases.
            catA = concatenate([*make_drop(convedAx), *make_drop(convedBx)])
            catB = concatenate([*make_drop(convedBx), *make_drop(convedAx)])
            (_, _, num_channels) = catA.shape
            (_, _, num_channelsB) = catB.shape
            assert tuple(catA.shape) == tuple(catB.shape), (catA.shape, catB.shape)

            width = 1 + 2*random.randrange(5, 8)
            size = random.randrange(23, 2*46)
            resNet = makeResNet(i, num_channels, width, size)

            convedAx.append(resNet(catA))
            convedBx.append(resNet(catB))

            assert len(convedAx) == len(convedBx), (len(convedAx), len(convedBx))
            for j, (a, b) in enumerate(zip(convedAx, convedBx)):
                assert tuple(a.shape) == tuple(b.shape), (i, j, a.shape, b.shape)
        return convedAx, convedBx

    convedA, convedB = make_block(embeddedA, embeddedB)
    # assert tuple(convedA.shape) == tuple(convedB.shape), (convedA.shape, convedB.shape)

    # TODO: check whether final BatchNorm would help?  (Pre dropout, of course.)
    # TODO: Similar for relu?
    # TODO: Try final Dropout with my other approaches, too.
    # TODO: Try different amounts of final dropout.  Can even try very high amounts, because we have so many dimensions at the end.
    # Approx 1,246 dimensions at the end for something close to `faithful` repro.
    # So could try even 90% dropout.
    make_end = Conv1D(name="output", filters=46, kernel_size=1, padding="same", strides=1, dtype='float32', kernel_initializer=msra)
    totes_clear = Layer(name='clear', dtype='float32')(make_end(SpatialDropout1D(rate=0)(concatenate(make_drop(convedA)))))
    totes_key = Layer(name='key', dtype='float32')(make_end(SpatialDropout1D(rate=0)(concatenate(make_drop(convedB)))))

    model = Model([inputA, inputB], [totes_clear, totes_key])

    model.compile(
        optimizer=tf.optimizers.Adam(),
        # optimizer=tf.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0), # momentum=0.9, nesterov=True),
        # optimizer=tfa.optimizers.AdamW(),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss_weights={'clear': 1/2, 'key': 1/2},
        metrics=[error],
    )
    return model

l = 100
hparams = {
    HP_DROPOUT: 0.0,
    HP_HEIGHT: 20,
    HP_blocks: 1,
    HP_bottleneck: 46 * 5,
    ## Idea: skip the first few short columns in the fractal.
    # HP_SKIP_HEIGH: 3,
    HP_WINDOW: l,
    HP_resSize: 0,
    HP_blowup: 1,
    HP_max_kernel: 1 + 2*6,
}

weights_name = "t5.h5"

make_model = make_model_recreate

def show():
    make_model(hparams).summary()

def main():
    # TODO: Actually set stuff to float16 only, in inference too.  Should use
    # less memory.
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    with tf.device(device_name):
        text = clean(load())
        # mtext = tf.convert_to_tensor(text)
        mtext = tf.convert_to_tensor(text)

        # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = "logs/scalars/{}".format(weights_name)
        tensorboard_callback = TensorBoard(log_dir=logdir, update_freq=50_000, profile_batch=0)

        checkpoint = ModelCheckpoint('weights/'+weights_name, monitor='loss', verbose=1, save_best_only=True)

        def schedule(epoch):
            # TODO: Aso try SGD with momentum.
            default = 0.001
            maxLR = 32 / 3
            lastEpoch = 12
            if epoch < lastEpoch:
                lr = default * min(maxLR, 1 + epoch * maxLR / 4)
            else:
                lr = default * max(1, maxLR * (1 - (epoch - lastEpoch) / 20))
            # NOTE: 32 was still fine, 64 broke.
            print(f"Scheduled learning rate for epoch {epoch}: {default} * {lr/default}")
            return lr

        def scheduleRampSGD(epoch):
            # TODO: Aso try SGD with momentum.
            default = 0.005
            # lr = default * (epoch / 2) **2
            lr = default * (epoch + 1)
            # NOTE: 32 was still fine, 64 broke.
            print(f"Scheduled learning rate for epoch {epoch}: {default} * {lr/default}")
            return lr
        def slow(epoch):
            return 0.0
            return 0.001 / 100



        callbacks_list = [
            checkpoint,
            tensorboard_callback,
            # hp.KerasCallback(logdir, hparams),
            # ReduceLROnPlateau(monitor='loss', mode='min', patience=5, cooldown=10, factor=1/2, verbose=1, min_delta=0.001),
            LearningRateScheduler(slow),
            EarlyStopping(monitor='loss', patience=60, verbose=1, restore_best_weights=True)
        ]

        with tf.summary.create_file_writer("logs/scalars").as_default():
            hp.hparams_config(
                hparams=[HP_DROPOUT, HP_HEIGHT, HP_WINDOW], metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
            )

        # try:
        #     print("Making model.")
        #     model = make_model(hparams)
        #     try:
        #         print("Trying to load weights.")
        #         model.load_weights('weights/'+weights_name)
        #     except:
        #         print("Loaded weights.")
        # except:
        model = keras.models.load_model('weights/'+weights_name)
        model.summary()
        print("Loaded model.")
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
                    x=TwoTimePadSequence(l, 10 ** 5 // 32, mtext, both=True),
                    # x = x, y = y,
                    # steps_per_epoch=10 ** 4 // 32,
                    max_queue_size=10**3,
                    initial_epoch=0,
                    # epochs=epoch+1,
                    # validation_split=0.1,
                    validation_data=TwoTimePadSequence(l, 10 ** 3 // 32, mtext, both=True),
                    epochs=100_000,
                    callbacks=callbacks_list,
                    # batch_size=batch_size,
                    verbose=1,
                    # workers=8,
                    # use_multiprocessing=True,
                    )
            except:
                print("Saving model...")
                model.save('weights/'+weights_name, include_optimizer=True)
                print("Saved model.")
                raise

        # (ciphers_t, labels_t, keys_t) = samples(text, 1000, l)
        # print("Eval:")
        # model.evaluate(TwoTimePadSequence(l, 10**4))

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
    main()
