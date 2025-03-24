#!/bin/bash

# This takes about 4 minutes for my 16 GiB corpus.
time cat ../corpus.txt | /usr/bin/time cargo run --release -- to-indices | pv > ../corpus.bytes

# check via:
head -c 1000 ../corpus.bytes | cargo run --release -- to-text 
