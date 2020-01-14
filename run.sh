#!/bin/bash
set -ex
NAME="$(grep '^weights_name *=' two_time_pad.py | grep -o '".*"')"
mkdir -p "runs/${NAME}"
cp two_time_pad.py  "runs/${NAME}/"

git add "runs/${NAME}"
git commit --allow-empty -m "Run: ${NAME}" two_time_pad.py "runs/${NAME}"
docker exec -it mystifying_germain python3 two_time_pad.py "${NAME}"
