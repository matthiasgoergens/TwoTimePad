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

def gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    else:
        raise NotImplementedError("Want GPU to limit memory growth.")

def limit_memory():
    from keras import backend as K
    sess = K.get_session()

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
from tensorboard.plugins.hparams import api as hp

from tensorflow.keras.layers import Embedding, Input, Dense, Dropout, Softmax, GlobalMaxPooling1D, MaxPooling1D, Conv1D, Flatten, concatenate, Bidirectional, LSTM, SimpleRNN, SeparableConv1D, TimeDistributed, BatchNormalization, SpatialDropout1D
from tensorflow_addons.layers import Maxout, Sparsemax
from tensorflow.keras.models import Sequential, Model

batch_size = 32

text = clean(load())
mtext = tf.convert_to_tensor(text, dtype='int8')
                 
def round_to(x, n):
  return (x // n) * n

def make1(window, text):
  (size,) = text.shape
  start = random.randrange(window)
  return tf.reshape(tf.slice(text, [start], [round_to(size - window, window)]), (-1, window))

mtext = tf.convert_to_tensor(text)

def makeEpochs(window):
  while True:
      x = make1(window, mtext)
      y = make1(window, mtext)
      for i in range(100):
        xx = tf.random.shuffle(x)
        yy = tf.random.shuffle(y)
        yield ((xx-yy)%46, (xx, yy))

class TwoTimePadSequence(keras.utils.Sequence):
  def _load(self):
      pass
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
                args=(num_samples,)
            )

def cipher_for_predict():
    # remove eol
    c1 = clean(open('TwoTimePad/examples/ciphertext-1.txt', 'r').read().lower()[:-1])
    # print(c1)
    c2 = clean(open('TwoTimePad/examples/ciphertext-2.txt', 'r').read().lower()[:-1])
    # print(c2)
    return tf.convert_to_tensor([sub(c1, c2)])


HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.5))
HP_HEIGHT = hp.HParam('height', hp.IntInterval(0, 30))
HP_WINDOW = hp.HParam('window', hp.IntInterval(1, 100))
HP_resSize = hp.HParam('resSize', hp.IntInterval(46, 8*46))

METRIC_ACCURACY = 'accuracy'

def make_model(hparams):
    n = hparams[HP_WINDOW]
    # my_input = Input(shape=(n,), dtype='int32', name="ciphertext")
    inputA = Input(shape=(n,), name="ciphertext")
    inputB = -inputA % 46
    resSize = hparams[HP_resSize]

    embedding = Embedding(output_dim=resSize, input_dim=len(alpha), name="my_embedding",
        batch_input_shape=[batch_size, n],
      )

    embeddedA = embedding(inputA)
    embeddedB = embedding(inputB)
    make_end = Conv1D(
        name='output',
        filters=46, kernel_size=1,
        padding='same', strides=1,
       )

    # clears = [make_end(embedded)]
    # keys = [make_end(embedded)]

    ## Best loss without conv at all was 4.5
    ## With one conv we are getting validation loss of 3.5996 quickly (at window size 30); best was ~3.3
    ## So adding another conv and lstm. val loss after first epoch about 3.3

    convedA = embeddedA
    convedB = embeddedB

    # Ideas: more nodes, no/lower dropout, only look for last layer for final loss.
    # nine layers is most likely overkill.
    def makeResNet(i):
      resInputMe = Input(name = 'res_inputMe', shape=(n,resSize,))
      resInputDu = Input(name = 'res_inputDu', shape=(n,resSize,))
      convedBroad = (
          TimeDistributed(BatchNormalization())(
          TimeDistributed(relu())(
          Conv1D(
            filters=4*46, kernel_size=15, padding='same',
            kernel_initializer=keras.initializers.he_normal(seed=None),
            )(
          TimeDistributed(BatchNormalization())(
          TimeDistributed(relu())(
          Conv1D(
            filters=3*46, kernel_size=1,
            kernel_initializer=keras.initializers.he_normal(seed=None),
            )(
          # TimeDistributed(keras.layers.AlphaDropout(0.5))(
          ( # SpatialDropout1D(rate=hparams[HP_DROPOUT])(
            resInputMe))))))))
      convedNarrow = (
          TimeDistributed(BatchNormalization())(
          TimeDistributed(relu())(
          Conv1D(
            filters=4*46, kernel_size=5, padding='same',
            kernel_initializer=keras.initializers.he_normal(seed=None),
            )(
          TimeDistributed(BatchNormalization())(
          TimeDistributed(relu())(
          Conv1D(
            filters=3*46, kernel_size=1,
            kernel_initializer=keras.initializers.he_normal(seed=None),
            )(
          # TimeDistributed(keras.layers.AlphaDropout(0.5))(
          ( # SpatialDropout1D(rate=hparams[HP_DROPOUT])(
            resInputMe))))))))
      innerConved =(
        keras.layers.Add(name='resOutput')([resInputMe,
          TimeDistributed(BatchNormalization())(
          TimeDistributed(relu())(
          Conv1D(filters=resSize, kernel_size=1,
            kernel_initializer=keras.initializers.he_normal(seed=None),
          )(
          ( # SpatialDropout1D(rate=hparams[HP_DROPOUT])(
          concatenate([resInputMe, convedNarrow, convedBroad, resInputDu])))))]))
      return Model([resInputMe, resInputDu], [innerConved], name=f'resnet{i}')

    for i in range(hparams[HP_HEIGHT]):
        resNet = makeResNet(i)
        convedA, convedB = resNet([convedA, convedB]), resNet([convedB, convedA])

    totes_clear = make_end(
        SpatialDropout1D(rate=hparams[HP_DROPOUT])(
            convedA))
    totes_key = make_end(
        SpatialDropout1D(rate=hparams[HP_DROPOUT])(
            convedB))

    model = Model([inputA], [totes_clear, totes_key])

    model.compile(
      optimizer=tf.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    return model

l = 60
hparams = {
    HP_DROPOUT: 0.1,
    HP_HEIGHT: 7,
    HP_WINDOW: l,
    HP_resSize: 8 * 46,
}

weights_name = 'recreate-best-faster-output.h5'

from datetime import datetime
from keras.callbacks import *
def main():
    gpu_memory_growth()

    # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f"logs/scalars/{weights_name}"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=5,  write_images=True, embeddings_freq=5)


    checkpoint = ModelCheckpoint(weights_name, verbose=1, save_best_only=True)

    callbacks_list = [checkpoint,
                      tensorboard_callback,
                      hp.KerasCallback(logdir, hparams),
                      keras.callbacks.ReduceLROnPlateau(patience=20, factor=0.5, verbose=1, min_delta=0.0001),
                      # keras.callbacks.EarlyStopping(patience=100, verbose=1, restore_best_weights=True)
                      ]

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
      hp.hparams_config(
        hparams=[HP_DROPOUT, HP_HEIGHT, HP_WINDOW],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
      )



    with tf.device(device_name):
      try:
        model = keras.models.load_model(weights_name)
        print("Loaded weights.")
      except:
        model = make_model(hparams)
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


      # model.evaluate(TwoTimePadSequence(l, 10**4 // 32), callbacks=[tensorboard_callback])
      # print("Training:")
      # (ciphers, labels, keys) = samples(text, training_size, l)
      # print(model.fit(ciphers, [labels, keys],
      for epoch, (x, y) in enumerate(makeEpochs(l)):
          print(f"My epoch: {epoch}")
          model.fit(x=x,
                    y=y,
                    # max_queue_size=1_000,
                    # initial_epoch=epoch,
                    # epochs=epoch+1,
                    validation_split=0.05,
                    callbacks=callbacks_list,
                    batch_size=32,
                    verbose=1,
                    # workers=8,
                    # use_multiprocessing=True,
                    )
          #(ciphers_t, labels_t, keys_t) = samples(text, 1000, l)
      #print("Eval:")
      #model.evaluate(TwoTimePadSequence(l, 10**4))

    # Idea: we don't need the full 50% dropout regularization, because our input is already random.
    # So try eg keeping 90% of units?  Just enough to punish big gross / small nettto co-adaptions.

    # But wow, this bigger network (twice as large as before) trains really well without dropout.  And no learning rate reduction, yet.
    # It's plateau-ing about ~2.54 loss at default learning rate after ~20 epoch.  (If I didn't miss a restart.)

if __name__=='__main__':
    main()
