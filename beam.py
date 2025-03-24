# First: prepare the model.  And prepare some data.
import tensorflow as tf
import numpy as np

from pprint import pp

from process_corpus import alpha

def setup():
  np.set_printoptions(precision=4)

  device_name = tf.test.gpu_device_name()
  if device_name != "/device:GPU:0":
      useGPU = False
      print(SystemError("GPU device not found", device_name))
      raise NotImplementedError("Want GPU")
  else:
      useGPU = True
      print("Found GPU at: {}".format(device_name))

  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def main():
    setup()
    # File path for your 16 GiB corpus file (replace with your actual file path)
    filename = 'corpus.bytes'

    # Read the file one byte at a time.
    # FixedLengthRecordDataset will yield records of length 1.
    dataset = tf.data.FixedLengthRecordDataset(filename, record_bytes=1)

    # Decode the raw bytes into uint8 values.
    # Each element in the dataset will be a scalar representing a byte.
    dataset = dataset.map(lambda x: tf.io.decode_raw(x, tf.uint8)[0])

    # Define the window size.
    # We want 10 input bytes and 1 target byte.
    window_size = 10
    total_window_size = window_size + 1

    # Create sliding windows of total_window_size with a stride of 1.
    # drop_remainder=True ensures every window has exactly total_window_size elements.
    windowed_dataset = dataset.window(total_window_size, shift=1, drop_remainder=True)
    windowed_dataset = windowed_dataset.flat_map(lambda window: window.batch(total_window_size))

    # Split each window into (input, target) where input is the first 10 bytes and target is the 11th.
    def split_input_target(window):
        input_bytes = window[:-1]
        target_byte = window[-1]
        return input_bytes, target_byte

    dataset = windowed_dataset.map(split_input_target)

    # (Optional) Shuffle and batch the dataset.
    # Adjust the shuffle buffer size and batch size according to your hardware.
    dataset = dataset.shuffle(10000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

    # Build a simple model.
    # We assume 256 possible byte values.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size,)),  # 10 input bytes
        tf.keras.layers.Embedding(input_dim=len(alpha), output_dim=len(alpha)),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(len(alpha), activation='softmax')  # Predict next byte as one of 256 classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Start training.
    # Note: Depending on the size of your dataset, you might need to adjust steps_per_epoch.
    model.fit(dataset, epochs=10)

if __name__ == "__main__":
    main()
