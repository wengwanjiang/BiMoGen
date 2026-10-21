#!/usr/bin/env bash
set -e

accelerate launch --multi_gpu --num_processes 8 -m training.train --stage pretrain
accelerate launch --multi_gpu --num_processes 8 -m training.train --stage sft \
  --resume output/bimogen/pretrain-final
