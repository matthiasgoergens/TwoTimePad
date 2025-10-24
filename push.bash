#!/usr/bin/env bash
set -euo pipefail

buckets=100
tmp=$(mktemp --directory)

find data -type f | sort --random-sort | \
  split --numeric-suffixes --number=l/"$buckets" --additional-suffix=.lst - "$tmp/batch-"

n=0
for chunk in "$tmp"/batch-*.lst; do
  xargs --no-run-if-empty --delimiter=$'\n' --arg-file="$chunk" git add --
  n=$((n+1))
  git commit --allow-empty --message="Add data batch $n/$buckets"
  git push github
done
