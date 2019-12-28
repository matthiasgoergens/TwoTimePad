# -*- coding: utf-8 -*-

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  useGPU = False
  print(SystemError('GPU device not found', device_name))
  raise NotImplementedError("Want GPU")
else:
  useGPU = True
  print('Found GPU at: {}'.format(device_name))

# Mixed precision
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)
###

import sys
import itertools as it

import re

import tensorflow
import tensorflow as tf
from tensorflow import keras


import numpy as np
import random
from pprint import pprint
import functools as ft

np.set_printoptions(precision=4)

alpha =    " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()".lower()
alphaRE = alpha.replace("-","\\-")

assert len(alpha) == 46

def load():
  # text = ' '.join(f.open('r').read() for f in pathlib.Path('data').glob('*.txt')).lower()
  text = open('corpus.txt', 'r').read().lower()
  text = re.sub('\s+', ' ', text)
  # text = re.sub(f'[^{alphaRE}]', '', text)
  text = re.sub('[^%s]' % alphaRE, '', text)
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
    ciphers.append(cipher)
    labels.append(label)
    keys.append(key)
  one_hot_ciphers = tf.convert_to_tensor(ciphers)
  one_hot_labels = tf.convert_to_tensor(labels)
  one_hot_keys = tf.convert_to_tensor(keys)
  return (one_hot_ciphers,
        one_hot_labels,
        one_hot_keys)

# relu = ft.partial(keras.layers.LeakyReLU, alpha=0.1)
relu = tf.keras.layers.PReLU

import tensorflow_addons as tfa

from tensorflow.keras.layers import Embedding, Input, Dense, Dropout, Softmax, GlobalMaxPooling1D, MaxPooling1D, Conv1D, Flatten, concatenate, Bidirectional, LSTM, SimpleRNN, SeparableConv1D, TimeDistributed, BatchNormalization, SpatialDropout1D
from tensorflow_addons.layers import Maxout, Sparsemax
from tensorflow.keras.models import Sequential, Model

batch_size = 64

text = clean(load())
                 
class TwoTimePadSequence(keras.utils.Sequence):
  def on_epoch_end(self):
    print("Epoch {self.epochs} ended.")
    self.epochs += 1
  def __len__(self):
    return self.training_size // batch_size
  def __getitem__(self, idx):
    (ciphers_p, labels_p, keys_p) = samples(text, batch_size, self.window)
    return ciphers_p, [labels_p, keys_p]
  def __init__(self, window, training_size):
    self.epochs = 0
    self.training_size =  (training_size // batch_size) * batch_size
    self.window = window

def cipher_for_predict():
    # remove eol
    c1 = clean(open('TwoTimePad/examples/ciphertext-1.txt', 'r').read().lower()[:-1])
    # print(c1)
    c2 = clean(open('TwoTimePad/examples/ciphertext-2.txt', 'r').read().lower()[:-1])
    # print(c2)
    return tf.convert_to_tensor([sub(c1, c2)])


# Idea: encode symmetry
#
# Provide both input and -input, then use shared weights (and mixing) to arrive
# at the two residuals..


#def make_2_model(n):
#    inputP = Input(shape=(n,), name="Piphertext")
#    inputN = -inputP % 46
#
#    resSize = 3 * 46
#
#    embedding = Embedding(output_dim=resSize, input_dim=len(alpha), name="my_embedding",
#        batch_input_shape=[batch_size, n])
#
#    convP = embedding(inputP)
#    convN = embedding(inputN)
#
#    make_end = Conv1D(
#            filters=46, kernel_size=1,
#            padding='same', strides=1,
#            kernel_initializer=keras.initializers.he_normal(seed=None),
#            # dtype=mixed_precision.Policy('float32')
#           )
    # (-1) % 46


def make_model(n):    
    # my_input = Input(shape=(n,), dtype='int32', name="ciphertext")
    my_input = Input(shape=(n,), name="ciphertext")
    resSize = 2 * 2 * 46
    embedded = (
      (
      Embedding(output_dim=resSize, input_dim=len(alpha), name="my_embedding",
        batch_input_shape=[batch_size, n],
      )(
      my_input
    )))

    def make_end(c, name):
        return ( # TimeDistributed(BatchNormalization())(
          Conv1D(
            name=name,
            filters=46, kernel_size=1,
            padding='same', strides=1,
            kernel_initializer=keras.initializers.he_normal(seed=None),
            # dtype=mixed_precision.Policy('float32')
           )(c))

    # clears = [make_end(embedded)]
    # keys = [make_end(embedded)]

    ## Best loss without conv at all was 4.5
    ## With one conv we are getting validation loss of 3.5996 quickly (at window size 30); best was ~3.3
    ## So adding another conv and lstm. val loss after first epoch about 3.3

    conved = embedded

    # Ideas: more nodes, no/lower dropout, only look for last layer for final loss.
    # nine layers is most likely overkill.
    for i in range(10):
      convedBroad = (
          TimeDistributed(relu())(
          TimeDistributed(BatchNormalization())(
          Conv1D(
            filters=2*46, kernel_size=15, padding='same',
            kernel_initializer=keras.initializers.he_normal(seed=None),
            )(
          TimeDistributed(relu())(
          TimeDistributed(BatchNormalization())(
          Conv1D(
            filters=2*46, kernel_size=1,
            kernel_initializer=keras.initializers.he_normal(seed=None),
            )(
          ( # TimeDistributed(keras.layers.AlphaDropout(0.5))(
            conved))))))))
      convedNarrow = (
          TimeDistributed(relu())(
          TimeDistributed(BatchNormalization())(
          Conv1D(
            filters=2*46, kernel_size=5, padding='same',
            kernel_initializer=keras.initializers.he_normal(seed=None),
            )(
          TimeDistributed(relu())(
          TimeDistributed(BatchNormalization())(
          Conv1D(
            filters=2*46, kernel_size=1,
            kernel_initializer=keras.initializers.he_normal(seed=None),
            )(
          ( # TimeDistributed(keras.layers.AlphaDropout(0.5))(
            conved))))))))
      conved =(
        keras.layers.Add()([conved,
          TimeDistributed(relu())(
          TimeDistributed(BatchNormalization())(
          Conv1D(filters=resSize, kernel_size=1,
            kernel_initializer=keras.initializers.he_normal(seed=None),
          )(
          concatenate([conved, convedNarrow, convedBroad]))))]))

    totes_clear = make_end(
        SpatialDropout1D(rate=1/2)(
            conved), name='clear')
    totes_key = make_end(
        SpatialDropout1D(rate=1/2)(
            conved), name='key')

    model = Model([my_input], [totes_clear, totes_key])

    model.compile(
      # optimizer=keras.optimizers.Adam(learning_rate=0.001 / 2**0),
      optimizer=tf.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      # TODO: write custom loss that uses 'from logits'.
      # loss=TimeDistributed(tf.nn.sparse_softmax_cross_entropy_with_logits),
      metrics=['accuracy'])
    return model

weights_name = 'thin10-dropout-logit.h5'
from datetime import datetime
# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/scalars/{weights_name}"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=5,  write_images=True, embeddings_freq=5)

from keras.callbacks import *
checkpoint = ModelCheckpoint(weights_name, verbose=1, save_best_only=True)

callbacks_list = [checkpoint,
                  tensorboard_callback,
                  # keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1, min_delta=0.0001),
                  # keras.callbacks.EarlyStopping(patience=100, verbose=1, restore_best_weights=True)
                  ]

l = 60
with tf.device(device_name):
  layers = 9
  try:
    model = keras.models.load_model(weights_name)
    print("Loaded weights.")
  except:
    model = make_model(l)
    model.summary()
    print("Failed to load weights.")
    # raise
  #for i in range(10*(layers+1)):

  print("Predict:")
  predict_size = 10
  
  # ita_cipher = cipher_for_predict()
  # [ita_label, ita_key] = model .predict(ita_cipher)
  # print(toChars_labels(ita_cipher))
  #pprint(toChars(ita_label)[:1000])
  #pprint(toChars(ita_key)[:1000])

  (ciphers_p, labels_p, keys_p) = samples(text, predict_size, l)
  [pred_label, pred_key] = model.predict(ciphers_p)
  # clear, key, prediction
  pprint(list(zip(
      toChars_labels(ciphers_p),

      toChars_labels(labels_p),
      toChars(pred_label),

      toChars_labels(keys_p),
      toChars(pred_key),

      predict_size * [l * " "]
      )),
    # width=250
    )


  print("text size: {:,}\tlayers: {}".format(len(text), layers))
  print("Window length: {}".format(l))

  model.evaluate(TwoTimePadSequence(l, 2*10**4), callbacks=[tensorboard_callback])
  # print("Training:")
  # (ciphers, labels, keys) = samples(text, training_size, l)
  # print(model.fit(ciphers, [labels, keys],
  print(model.fit(x=TwoTimePadSequence(l, 10**4),
            max_queue_size=10_000,
            # initial_epoch=27,
            epochs=1000+layers, # Excessively long.  But early stopping should rescue us.
            validation_data=TwoTimePadSequence(l, 2*10**3),
            callbacks=callbacks_list))
  #(ciphers_t, labels_t, keys_t) = samples(text, 1000, l)
  #print("Eval:")
  #model.evaluate(TwoTimePadSequence(l, 10**4))
  model.save("{}_layers_{}".format(weights_name, layers))

# Idea: we don't need the full 50% dropout regularization, because our input is already random.
# So try eg keeping 90% of units?  Just enough to punish big gross / small nettto co-adaptions.

# But wow, this bigger network (twice as large as before) trains really well without dropout.  And no learning rate reduction, yet.
# It's plateau-ing about ~2.54 loss at default learning rate after ~20 epoch.  (If I didn't miss a restart.)
