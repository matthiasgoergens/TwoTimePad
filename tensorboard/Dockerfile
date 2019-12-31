# Latest python version with a working tensorflow?
# FROM python:3.7
FROM tensorflow/tensorflow:latest-py3

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y bash-completion vim git
RUN pip3 install tensorflow_addons

WORKDIR /app
# CMD sleep infinity
CMD whoami && tensorboard --port 80 --bind_all --logdir logs/
