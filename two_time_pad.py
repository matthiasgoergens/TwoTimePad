# -*- coding: utf-8 -*-
"""two time pad

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kFhZ4ar-iCZ6t8MwW6jThk4RMKWFVcJy
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

import pathlib
import re

import tensorflow
import tensorflow as tf
from tensorflow import keras


import pathlib
import numpy as np
import random
from pprint import pprint

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

alpha =    " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()".lower()
alphaRE = alpha.replace("-","\\-")

assert len(alpha) == 46

def load():
  text = ' '.join(f.open('r').read() for f in pathlib.Path('data').glob('*.txt')).lower()
  text = re.sub('\s+', ' ', text)
  text = re.sub(f'[^{alphaRE}]', '', text)
  return text

def add(clear, key):
  return [(a+b) % len(alpha) for a, b in zip(clear, key)]

def sub(cipher, key):
  return [(a-b) % len(alpha) for a, b in zip(cipher, key)]

def clean(text):
  t = {c: i for i, c in enumerate(alpha)}
  return [t[c] for c in text]

def toChar(numbers):
  return ''.join(alpha[i] for i in numbers)

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
  return text[start:start+l]

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
  return (one_hot_ciphers,
        one_hot_labels,
        one_hot_keys)

# relu = keras.activations.relu # (alpha=0.1)
relu = lambda: keras.layers.LeakyReLU(alpha=0.1)
# keras.layers.Embedding

from tensorflow.keras.layers import Embedding, Input, Dense, Dropout, Softmax, GlobalMaxPooling1D, MaxPooling1D, Conv1D, Flatten, concatenate, Bidirectional, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential, Model
import tensorflow_addons as tfa

Maxout = tfa.layers.Maxout
# window_size = 200
# TODO: use convolution over 1d.
# tf.keras.layers.Conv1D
def make_model(n):
  # Not sure if lambda is necessary to not share weights?
  dropout = lambda: Dropout(rate=0.1)
  
  my_input = Input(shape=(n,), dtype='int32', name="ciphertext")
  embedded = (
    dropout()(
    # Give larger embedding to allow for dropout?
    Embedding(output_dim=2*len(alpha), input_dim=len(alpha), name="my_embedding")(
    my_input
  )))

  # Idea is to bubble up better and better guesses for what letters we have locally.
  conv = lambda base: Maxout(46*2*2)(dropout()(Conv1D(
      filters=46*2*2 * 5, kernel_size=5,
      padding='same', strides=1)(
        base
      )))
  # Not sure whether the eta abstraction is necessary?
  lstm = lambda base: (
    Bidirectional(LSTM(46*2*2, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(
      base))
  relud = lambda base: relu()(dropout()(Conv1D(
      filters=46*2*2 * 5, kernel_size=5,
      padding='same', strides=1)(
        base
    )))
  lstm_ed = (
    relud(conv(
    lstm(conv(
    relud(conv(
    lstm(conv(
    relud(conv(
    embedded
  )))))))))))

  totes_clear = (
    Softmax(name="clear")(Conv1D(
      filters=len(alpha), kernel_size=5,
      padding='same', strides=1)(
    lstm_ed
  )))
  totes_key = (
    Softmax(name="key")(Conv1D(
      filters=len(alpha), kernel_size=5,
      padding='same', strides=1)(
    lstm_ed
  )))
  model = Model([my_input], [totes_clear, totes_key])

  model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
  return model

import sys

training_size = 2 * 10**4
test_size = 10**3


from datetime import datetime
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# with tf.device('/device:GPU:0'):
# from google.colab import drive
# drive.mount('/content/drive')
weights_name = 'my_model_weights-increasing-sub-local-relu.h5'

# Starting from higher.
import sys
# l = int(sys.argv[1])
l = 100
# l = window_size

l = 1 # Already trained


while True:
  model = make_model(l)
  try:
    model.load_weights(weights_name)
    print("Loaded weights.")
  except:
    print("Failed to load weights.")

  print(model)
  print()
  text = clean(load())
  print(f"text size: {len(text):,}")
  print(f"Window length: {l}")

  print("Predict:")
  predict_size = 5
  (ciphers_p, labels_p, keys_p) = samples(text, predict_size, l)
  [pred_label, pred_key] = (model.predict(ciphers_p))
  # clear, key, prediction
  pprint(list(zip(
      #toChars_labels(ciphers_p),
      #toChars(pred_cipher),
      # predict_size * [l * " "],

      toChars_labels(labels_p),
      toChars(pred_label),
      predict_size * [l * " "],

      toChars_labels(keys_p),
      toChars(pred_key),
      predict_size * [l * " "]
      )),
    width=250)
  print(f"Window length: {l}")
  print("Training:")
  (ciphers, labels, keys) = samples(text, training_size, l)
  model.fit(ciphers, [labels, keys],
            validation_split=1/16,
            callbacks=[tensorboard_callback],)
  l = min(200, l+1)
  model.save_weights(weights_name)
  print("Saved weights")


# before relu change, with four lstm, best was:
# Window length: 49
# 20000/20000 [==============================] - 530s 26ms/sample - loss: 3.5863 - clear_loss: 1.7917 - key_loss: 1.7945 - clear_accuracy: 0.4630 - key_accuracy: 0.4618 - val_loss: 3.5398 - val_clear_loss: 1.7722 - val_key_loss: 1.7684 - val_clear_accuracy: 0.4733 - val_key_accuracy: 0.4723
