#!/usr/bin/env bash
# Reproduce everything (data download, tokenizer, training, evaluation, test predictions, model export)
# by executing the tutorial notebook top to bottom. Seeded (CONFIG["seed"] = 42).
#   ./run.sh            full run (several hours on an Apple M1; much faster on a CUDA GPU)
#   MT_DEBUG=1 ./run.sh smoke test on 20k pairs / 150 steps (a few minutes)
set -euo pipefail
cd "$(dirname "$0")"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 fr_en_transformer.ipynb
