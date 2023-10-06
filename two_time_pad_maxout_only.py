# -*- coding: utf-8 -*-
"""two time pad - maxout only

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qen8Dv8Zcyen4IzEk3XuKavqitT35wax
"""

# !rm -rf TwoTimePad
# !git clone https://github.com/matthiasgoergens/TwoTimePad.git
# !rm -rf data *.h5
# !cp -r TwoTimePad/data TwoTimePad/*.h5 ./
# !pip install -q --no-deps tensorflow-addons

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

# device_name = tf.test.gpu_device_name()

# if device_name != '/device:GPU:0':
#   print(
#       '\n\nThis error most likely means that this notebook is not '
#       'configured to use a GPU.  Change this in Notebook Settings via the '
#       'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
#   raise SystemError('GPU device not found')

# with tf.device('/device:GPU:0'):
if True:
    import functools as ft
    import pathlib
    import random
    import re
    from pprint import pprint

    import numpy as np
    import tensorflow
    import tensorflow as tf
    from tensorflow import keras

    # Train with longer later.
    # n = 25
    # n = 50

    np.set_printoptions(precision=4)

    # Idea:
    # Load corpus, and clean, convert into numbers
    # Repeatedly:
    #   create a batch of data
    #   train
    #   Optional: show current results.

    alpha = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()".lower()
    alphaRE = alpha.replace("-", "\\-")

    assert len(alpha) == 46

    def load():
        text = " ".join(
            f.open("r").read() for f in pathlib.Path("data").glob("*.txt")
        ).lower()
        text = re.sub("\s+", " ", text)
        text = re.sub(f"[^{alphaRE}]", "", text)
        return text

    def add(clear, key):
        return [(a + b) % len(alpha) for a, b in zip(clear, key)]

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
                    [
                        (tensor[lineNum, cN, alphaN], alphaN)
                        for alphaN in range(alphaNum)
                    ]
                )
                chars.append(char)
            output.append(toChar(chars))
        return output

    def toChars_labels(labels):
        (linesNums, charNum) = labels.shape
        output = []
        for lineNum in range(linesNums):
            chars = []
            for cN in range(charNum):
                chars.append(labels[lineNum, cN])
            output.append(toChar(chars))
        return output

    def prepare(clear, key):
        assert len(clear) == len(key), (clear, key)

        depth = len(alpha)

        # label = clear[len(clear)//2:len(clear)//2+1]
        cipher = sub(clear, key)
        return (cipher, clear)

    def sample(text, l):
        start = random.randrange(len(text) - l)
        return text[start : start + l]

    def samples(text, batch_size, l):
        ciphers = []
        keys = []
        labels = []
        for _ in range(batch_size):
            clear = sample(text, l)
            key = sample(text, l)

            (cipher, label) = prepare(clear, key)
            # print(toChar(clear))
            # print(toChar(key))
            # print(toChar(add(clear, key)))
            # print(toChar(sub(add(clear, key), key)))
            ciphers.append(cipher)
            labels.append(label)
            keys.append(key)
        one_hot_ciphers = tf.convert_to_tensor(ciphers)
        one_hot_labels = tf.convert_to_tensor(labels)
        one_hot_keys = tf.convert_to_tensor(keys)
        # print(one_hot_ciphers.shape)
        # print(one_hot_labels.shape)
        # print(len(one_hot_labels))
        return (one_hot_ciphers, one_hot_labels, one_hot_keys)

    # relu = keras.activations.relu # (alpha=0.1)
    relu = ft.partial(keras.layers.LeakyReLU, alpha=0.1)
    # keras.layers.Embedding

    import tensorflow_addons as tfa
    from tensorflow.keras.layers import (
        LSTM,
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        Embedding,
        Flatten,
        GlobalMaxPooling1D,
        Input,
        MaxPooling1D,
        SimpleRNN,
        Softmax,
        concatenate,
    )
    from tensorflow.keras.models import Model, Sequential

    Maxout = tfa.layers.Maxout
    # window_size = 200
    # TODO: use convolution over 1d.
    # tf.keras.layers.Conv1D
    def make_model(n):
        # Not sure if lambda is necessary to not share weights?
        dropout = ft.partial(Dropout, rate=0.1)

        my_input = Input(shape=(n,), dtype="int32", name="ciphertext")
        embedded = dropout(name="input_dropout")(
            Embedding(output_dim=len(alpha), input_dim=len(alpha), name="my_embedding")(
                my_input
            )
        )

        # relud = lambda base: relu()(Dense()(base))
        # Idea is to bubble up better and better guesses for what letters we have locally.
        conv = lambda name, kernel_size=5, **kwargs: lambda base: (
            Maxout(200, name=name + "_maxout")(
                dropout(name=name + "_dropout")(
                    Conv1D(
                        name=name + "_conv",
                        filters=1000,
                        kernel_size=kernel_size,
                        padding="same",
                        strides=1,
                        **kwargs,
                    )(base)
                )
            )
        )

        ## Best loss without conv at all was 4.5
        ## With one conv we are getting validation loss of 3.5996 quickly (at window size 30); best was ~3.3
        ## So adding another conv and lstm. val loss after first epoch about 3.3

        conved = conv(name="4conv9", kernel_size=9)(
            conv(name="3conv9", kernel_size=9)(
                conv(name="2conv9", kernel_size=9)(
                    conv(name="1conv9", kernel_size=9)(embedded)
                )
            )
        )
        lstmed = (
            # lstm(name="2lstm")(
            # lstm(name="1_lstm")(
            conv(name="1convDilate_3", kernel_size=1)(
                conv(name="1convDilate_2", kernel_size=1)(
                    conv(name="1convDilate_1_18", kernel_size=18)(conved)
                )
            )
        )
        last_conv = conv(name="2cl5", kernel_size=5)(
            conv(name="1cl_dilate", kernel_size=1)(concatenate([conved, lstmed]))
        )

        gather_name = ""
        totes_clear = Conv1D(
            filters=len(alpha),
            kernel_size=1,
            padding="same",
            strides=1,
            name="clear" + gather_name,
            activation="softmax",
        )(last_conv)
        totes_key = Conv1D(
            filters=len(alpha),
            kernel_size=1,
            padding="same",
            strides=1,
            name="key" + gather_name,
            activation="softmax",
        )(last_conv)
        model = Model([my_input], [totes_clear, totes_key])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    import sys

    training_size = 2 * 10 ** 4
    # test_size = 2 * 10**3


from datetime import datetime

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

from keras.callbacks import ModelCheckpoint

weights_name = "layers-new-local.h5"
checkpoint = ModelCheckpoint(weights_name, verbose=1, save_best_only=False)

callbacks_list = [checkpoint, tensorboard_callback]

# with tf.device('/device:GPU:0'):
if True:
    # from google.colab import drive
    # drive.mount('/content/drive')

    # Starting from higher.
    import sys

    # l = int(sys.argv[1])
    l = 100
    # l = window_size

# l = 1 # Already trained on the shorter ones
import itertools as it

ls = it.chain(range(30, 50, 5), it.count(50))

# with tf.device('/device:GPU:0'):
if True:
    model = make_model(50)
    model.summary()
    while True:
        l = next(ls)
        model = make_model(l)
        try:
            model.load_weights(weights_name, by_name=True)
            print("Loaded weights.")
        except:
            print("Failed to load weights.")
            # raise

        print(model)
        print()
        text = clean(load())
        print(f"text size: {len(text):,}")
        print(f"Window length: {l}")

        print("Predict:")
        predict_size = 5
        (ciphers_p, labels_p, keys_p) = samples(text, predict_size, l)
        [pred_label, pred_key] = model.predict(ciphers_p)
        # clear, key, prediction
        pprint(
            list(
                zip(
                    # toChars_labels(ciphers_p),
                    # toChars(pred_cipher),
                    # predict_size * [l * " "],
                    toChars_labels(labels_p),
                    toChars(pred_label),
                    predict_size * [l * " "],
                    toChars_labels(keys_p),
                    toChars(pred_key),
                    predict_size * [l * " "],
                )
            ),
            width=250,
        )
        print(f"Window length: {l}")
        # (ciphers_t, labels_t, keys_t) = samples(text, test_size, l)
        # print("Eval:")
        # model.evaluate(ciphers_t,  [ciphers_t, labels_t, keys_t], verbose=2)

        print("Training:")
        (ciphers, labels, keys) = samples(text, training_size, l)
        model.fit(
            ciphers, [labels, keys], validation_split=1 / 16, callbacks=callbacks_list
        )
        # l = min(200, l+1)
        # model.save_weights(weights_name)
        # print("Saved weights")
