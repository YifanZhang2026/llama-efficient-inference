#!/bin/bash
# Reproduce the original pipeline: end-to-end benchmark, then train + eval both MoE modes.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
python -u run_benchmark.py                 > logs/benchmark_orig.log 2>&1
python -u train_moe.py --init-mode slice   > logs/train_slice.log 2>&1
python -u eval_moe.py  --init-mode slice   > logs/eval_slice.log 2>&1
cp checkpoints/moe_finetuned.pt checkpoints/moe_slice.pt
python -u train_moe.py --init-mode lora    > logs/train_lora.log 2>&1
python -u eval_moe.py  --init-mode lora    > logs/eval_lora.log 2>&1
cp checkpoints/moe_finetuned.pt checkpoints/moe_lora.pt
echo ALL DONE
