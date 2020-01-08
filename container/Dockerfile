# Latest python version with a working tensorflow?
# FROM python:3.7
FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y bash-completion vim git
RUN pip3 install tensorflow_addons
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ENV TF_XLA_FLAGS=--tf_xla_auto_jit=2

WORKDIR /app
# CMD sleep infinity
CMD whoami && sleep infinity
