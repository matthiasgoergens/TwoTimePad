#!/bin/bash
# sed -E \
#   -e 's/.*/\L&/' \
#   -e 's/[[:space:]]+/ /g' \
#   -e "s/[^ a-z0-9\.?,-:;'()]//g"

# sed -z -E 's/.*/\L&/; s/[[:space:]]+/ /g; s/[^- a-z0-9.?,:;'"'"'()]//g'

tr '[:upper:]' '[:lower:]' | \
tr '[:space:]' ' ' | \
tr -s ' ' | \
tr -cd " \\-a-z0-9.?,:;'()"
