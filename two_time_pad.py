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
    Add,
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
# from tensorflow_addons.layers import Maxout, Sparsemax

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
            return (self.cipherA[i, :, :], self.cipherB[i, :, :]), (self.aa[i, :, :], self.bb[i, :, :])
            # return (self.cipherA[i, :, :], ), (self.aa[i, :, :], self.bb[i, :, :])
            # return (self.cipherA[i, :, :], ), (self.aa[i, :, :],)

    def __init__(self, window, training_size, mtext):
        self.a = make1(window, mtext)
        self.b = make1(window, mtext)

        self.epochs = 0
        self.training_size = training_size
        self.window = window
        self._load()


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

def make_model_global_local(hparams):
    ic = lambda: Sequential([
        BatchNormalization(),
        SpatialDropout1D(rate=hparams[HP_DROPOUT]),
    ])

    n = hparams[HP_WINDOW]
    height = hparams[HP_HEIGHT]

    inputA = Input(shape=(n,), name="ciphertextA", dtype='int32')
    inputB = Input(shape=(n,), name="ciphertextB", dtype='int32')
    embedding = Embedding(
        output_dim=len(alpha), input_length=n, input_dim=len(alpha), name="my_embedding", batch_input_shape=[batch_size, n],)

    embeddedA = embedding(inputA)
    embeddedB = embedding(inputB)

    random.seed(23)
    def make_block(globalA, globalB):
        # width = 1 + 2 * 3
        # width = 1 + 2 * random.randrange(1, 5)
        width = random.randrange(1, 12)
        local_dims = 2*46
        more_global = 46

        num_layers = height

        post_conv1A, post_conv1B = None, None
        post_reluA, post_reluB = None, None
        post_icA, post_icB = None, None

        # Idea: keep track of local states and feed them into final layer, too.
        localA, localB = None, None
        localsA, localsB = [], []

        for i in range(num_layers):
            shapeIt = Sequential([
                # Idea: also investigate flipping relu and ic (both here and further down.)
                # Idea: look into maxout.
                relu(),
                ic(),
                Conv1D(filters=local_dims + 2*more_global, kernel_size=1, padding='same', kernel_initializer=msra),
            ])

            shapeItA, shapeItB = (
                shapeIt(concat([localA, localB, globalA, globalB])),
                shapeIt(concat([localB, localA, globalB, globalA])),
            )

            post_conv1A, post_conv1B = (
                plus(post_conv1A, shapeItA),
                plus(post_conv1B, shapeItB),
            )

            relu_1 = relu()
            post_reluA, post_reluB = (
                plus(post_reluA, relu_1(post_conv1A)),
                plus(post_reluB, relu_1(post_conv1B)),
            )

            icR = ic()
            post_icA, post_icB = (
                plus(post_icA, icR(post_reluA)),
                plus(post_icB, icR(post_reluB)),
            )

            # Idea: could cat A and B before feeding to the last two convs.
            # But would take more parameters.  (But symmetric.)
            # But could make up for this with fewer local_dims..
            # Idea: could also feed either new local state to new global state,
            # or vice versa.  But losing parallelism then.
            conv5 = Conv1D(filters=local_dims,  kernel_size=width, padding='same', kernel_initializer=msra)
            localA, localB = (
                plus(localA,  conv5(post_icA)),
                plus(localB,  conv5(post_icB)),
            )
            localsA.append(localA)
            localsB.append(localB)

            conv5G = Conv1D(filters=more_global, kernel_size=width, padding='same', kernel_initializer=msra)
            globalA, globalB = (
                cat(globalA, conv5G(post_icA)),
                cat(globalB, conv5G(post_icB)),
            )
        return (
            concat([globalA, *localsA]),
            concat([globalB, *localsB]),
        )

    random.seed(23)

    lastA, lastB = make_block(embeddedA, embeddedB)
    ## TODO: Idea for block design
    ## Bottleneck the state for the next block, but still pass the complete
    ## internal state of each bock onto the final pre-softmax layer.

    make_end = Sequential([
        relu(),
        ic(),
        Conv1D(name="output", filters=46, kernel_size=1, padding="same", strides=1, dtype='float32', kernel_initializer=msra),
    ], name='out')
    totes_clear = Layer(name='clear', dtype='float32')(make_end(lastA))
    totes_key = Layer(name='key', dtype='float32')(make_end(lastB))
    # Idea: Try a virtual totes_key, derived from totes_clear by a shift depending on cipher-text.
    # totes_key = make_end('key')(convedA)

    model = Model([inputA, inputB], [totes_clear, totes_key])
    opt = tf.optimizers.Adam()

    model.compile(
        optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"],
    )
    return model

l = 100
hparams = {
    HP_DROPOUT: 0.05,
    HP_HEIGHT: 10,
    HP_WINDOW: l,
    HP_resSize: 4 * 46,
}

weights_name = "glocal-10-_dropout-msra.h5"


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
        tensorboard_callback = TensorBoard(log_dir=logdir, update_freq=1_000, profile_batch=0, histogram_freq=5,  write_images=True, embeddings_freq=5)

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
            model = make_model_global_local(hparams)
            model.load_weights('weights/'+weights_name)
        except:
            try:
                model = keras.models.load_model('weights/'+weights_name)
                print("Loaded weights.")
            except:
                model = make_model_global_local(hparams)
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
                    x=TwoTimePadSequence(l, 10 ** 4 // 16, mtext),
                    # x = x, y = y,
                    # steps_per_epoch=10 ** 4 // 32,
                    max_queue_size=10**3,
                    # initial_epoch=311,
                    # epochs=epoch+1,
                    # validation_split=0.1,
                    # validation_data=TwoTimePadSequence(l, 2*10 ** 3 // 32, mtext),
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
