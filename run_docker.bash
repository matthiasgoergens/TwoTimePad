#!/bin/bash

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --rm nvcr.io/nvidia/tensorflow:25.02-tf2-py3 python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
