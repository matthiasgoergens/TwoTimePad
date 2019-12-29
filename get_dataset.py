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
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras.layers import (
    LSTM,
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
)
from tensorflow.keras.models import Model, Sequential

import tensorflow_addons as tfa
from keras.callbacks import *
from tensorflow_addons.layers import Maxout, Sparsemax

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
                [(tensor[lineNum, cN, alphaN], alphaN) for alphaN in range(alphaNum)]
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
        ciphers.append(cipher)
        labels.append(label)
        keys.append(key)
    one_hot_ciphers = tf.convert_to_tensor(ciphers)
    one_hot_labels = tf.convert_to_tensor(labels)
    one_hot_keys = tf.convert_to_tensor(keys)
    return (one_hot_ciphers, one_hot_labels, one_hot_keys)


batch_size = 32

text = clean(load())


class TwoTimePadSequence(keras.utils.Sequence):
    def on_epoch_end(self):
        print("Epoch {self.epochs} ended.")
        self.epochs += 1

    def __len__(self):
        return self.training_size

    def __getitem__(self, idx):
        (ciphers_p, labels_p, keys_p) = samples(text, batch_size, self.window)
        return ciphers_p, [labels_p, keys_p]

    def __init__(self, window, training_size):
        self.epochs = 0
        self.training_size = training_size
        self.window = window


def art():
    class ArtificialDataset(tf.data.Dataset):
        def _generator(num_samples):
            # Opening the file
            time.sleep(0.03)

            for sample_idx in range(num_samples):
                # Reading data (line, record) from the file
                time.sleep(0.015)

                yield (sample_idx,)

        def __new__(cls, num_samples=3):
            return tf.data.Dataset.from_generator(
                cls._generator,
                output_types=tf.dtypes.int64,
                output_shapes=(1,),
                args=(num_samples,),
            )


def cipher_for_predict():
    # remove eol
    c1 = clean(open("TwoTimePad/examples/ciphertext-1.txt", "r").read().lower()[:-1])
    # print(c1)
    c2 = clean(open("TwoTimePad/examples/ciphertext-2.txt", "r").read().lower()[:-1])
    # print(c2)
    return tf.convert_to_tensor([sub(c1, c2)])


l = 60


def main():
    pass


if __name__ == "__main__":
    main()
