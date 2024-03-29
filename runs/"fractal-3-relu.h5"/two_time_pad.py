# -*- coding: utf-8 -*-

import functools as ft
import itertools as it
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

            # yield (cipherX, cipherY), (
            #     xx,
            #     yy,
            # )

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
HP_WINDOW = hp.HParam("window", hp.IntInterval(1, 100))
HP_resSize = hp.HParam("resSize", hp.IntInterval(46, 8 * 46))

METRIC_ACCURACY = "accuracy"

relu = ft.partial(tf.keras.layers.PReLU, shared_axes=[1])

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
def cat(a, b):
    if a is None:
          return b
    elif b is None:
        return a
    else:
        return concatenate([a, b])

msra = tf.initializers.VarianceScaling(scale=2.0, distribution='truncated_normal')

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
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        # optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss_weights={'clear': 1/2, 'key': 1/2},
        metrics=[nAccuracy],
    )
    return model

def sequential(*layers):
    def helper(last):
        for layer in layers:
            last = layer(last)
        return last
    return helper

def make_model_fractal(hparams):
    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]
    ic = lambda: sequential(
        BatchNormalization(),
        SpatialDropout1D(rate=hparams[HP_DROPOUT]),
    )

    input = Input(shape=(n,), name="ciphertextA", dtype='int32')
    base = 3 * 46
    blowup = 3
    embedded = Embedding(
        output_dim=46, input_length=n, input_dim=len(alpha), name="embeddingA", batch_input_shape=[batch_size, n],)(
            input)

    # conved = Sequential([
    #     ic(),
    #     Conv1D(filters=base, kernel_size=9, padding='same', kernel_initializer=msra),
    # ])(embedded)
    # conved = embedded

    # Went from [x | blowup] * base to base to blowup * base
    # So could use cat?
    # Now: geting from base -> blowup * base -> base
    conv = lambda: sequential(
            # Idea: parallel of different kernel sizes.  Will save on trainable params.
            relu(),
            ic(),
            Conv1D(filters=base, kernel_size=9, padding='same', kernel_initializer=msra),
            )

    def block(n):
        if n <= 0:
            # Identity.  Work out whether we can/should use conv instead?
            # If we use conv, we can have embedding go to 46 instead of base, I think.
            return lambda x: x
        elif n <= 1:
            return conv()
        else:
            # f 0 = identity (or conv in paper)
            # f (n+1) = (f n . f n) + conv
            def helper(input):
                return average([
                    block(n-1)
                    (block(n-1)
                    (input)),

                    conv()
                    (input),
                ])
            return helper

    # Idea: Start first res from ic() or conv.
    # Idea: also give input directly, not just embedding?

    conved = block(height)(embedded)
    make_end = lambda name: sequential(
        # Maxout(base),
        # ic(),
        Conv1D(name=name, filters=46, kernel_size=1, padding="same", strides=1, dtype='float32', kernel_initializer=msra),
    )
    clear = make_end('clear')(conved)
    model = Model([input], [clear])

    model.compile(
        # optimizer=tf.optimizers.Adam(learning_rate=0.001),
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss_weights={'clear': 1/2, 'key': 1/2},
        metrics=[nAccuracy],
    )
    return model

l = 100
hparams = {
    HP_DROPOUT: 0.0,
    HP_HEIGHT: 3,
    HP_WINDOW: l,
    HP_resSize: 4 * 46,
}

weights_name = "fractal-3-relu.h5"

make_model = make_model_fractal

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
        tensorboard_callback = TensorBoard(log_dir=logdir, update_freq=10_000, profile_batch=0)

        checkpoint = ModelCheckpoint('weights/'+weights_name, monitor='loss', verbose=1, save_best_only=True)

        def schedule(epoch):
            endF = 0.2
            startF = 10 / endF
            steps = 50

            default = 0.001
            learning_rate= default * endF


            # As epoch goes from 0 to steps, startF goes from startF to 1
            sched = {steps-i: startF ** (i/steps) for i in reversed(range(1, steps+1))}
            lr = learning_rate * sched.get(epoch, 1)
            print(f"Scheduled learning rate for epoch {epoch}: {default} * {lr/default}")
            return lr

        callbacks_list = [
            checkpoint,
            tensorboard_callback,
            # hp.KerasCallback(logdir, hparams),
            ReduceLROnPlateau(monitor='loss', patience=3, cooldown=10, factor=1/2, verbose=1, min_delta=0.001),
            # LearningRateScheduler(schedule),
            EarlyStopping(monitor='loss', patience=30, verbose=1, restore_best_weights=True)
        ]

        with tf.summary.create_file_writer("logs/scalars").as_default():
            hp.hparams_config(
                hparams=[HP_DROPOUT, HP_HEIGHT, HP_WINDOW], metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
            )

        try:
            model = make_model(hparams)
            model.load_weights('weights/'+weights_name)
        except:
            try:
                model = keras.models.load_model('weights/'+weights_name)
                print("Loaded weights.")
            except:
                model = make_model(hparams)
                model.summary()
                print("Failed to load weights.")
                # raise
        # for i in range(10*(layers+1)):

        print("Predict:")
        predict_size = 10

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
                    x=TwoTimePadSequence(l, 10 ** 4 // 16, mtext, both=False),
                    # x = x, y = y,
                    # steps_per_epoch=10 ** 4 // 32,
                    max_queue_size=10**3,
                    # initial_epoch=311,
                    # epochs=epoch+1,
                    # validation_split=0.1,
                    validation_data=TwoTimePadSequence(l, 2*10 ** 3 // 32, mtext, both=False),
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


# Base loss for one side:
# log(46, 2)
# 5.523561956057013

if __name__ == "__main__":
    main()
