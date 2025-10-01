#!/bin/bash

# This takes about 4 minutes for my 16 GiB corpus.
time cat ../examples/ciphertext-1.txt | /usr/bin/time cargo run --release -- to-indices | pv > ../examples/ciphertext-1.bytes
time cat ../examples/ciphertext-2.txt | /usr/bin/time cargo run --release -- to-indices | pv > ../examples/ciphertext-2.bytes

