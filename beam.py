# First: prepare the model.  And prepare some data.
import tensorflow as tf
import numpy as np

from pprint import pp

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    useGPU = False
    print(SystemError("GPU device not found", device_name))
    raise NotImplementedError("Want GPU")
else:
    useGPU = True
    print("Found GPU at: {}".format(device_name))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
