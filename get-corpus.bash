#!/bin/bash
find data/ -name '*.txt' -print0 | xargs -0 -- cat > corpus.txt 
