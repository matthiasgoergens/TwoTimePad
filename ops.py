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
