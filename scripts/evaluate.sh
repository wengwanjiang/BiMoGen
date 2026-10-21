#!/usr/bin/env bash
set -e

python -m evaluation.eval_t2m --checkpoint checkpoints/bimogen-85000 \
  --seed 42 --output-json output/evaluation/t2m.json
python -m evaluation.eval_m2t --checkpoint checkpoints/bimogen-85000 \
  --seed 42 --output-json output/evaluation/m2t.json
