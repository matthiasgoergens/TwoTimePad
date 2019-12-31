#!/bin/bash
docker build -t matthias-nvidia .
docker run -it --gpus all --mount "type=bind,source=$(realpath ..),target=/app" matthias-nvidia sleep infinity
