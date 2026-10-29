#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
if [ ! -d data/raw/MedQuAD/.git ]; then
  git clone --depth=1 https://github.com/abachaa/MedQuAD.git data/raw/MedQuAD
fi
curl -fL https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json -o data/raw/ori_pqal.json
