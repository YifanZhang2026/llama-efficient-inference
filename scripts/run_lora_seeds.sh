#!/bin/bash
# LoRA MoE, 6 epochs, one train -> eval chain per seed, all in parallel on one
# GPU (each seed has its own checkpoint). Default seeds: 1 2 3 42. Seed 42
# repeats lora_ep6 from run_sweep.sh (run alone): if its losses / PPL / acc
# match bit for bit, running in parallel did not change the numerics.
# Epoch times and generation speeds in these logs are NOT comparable to other
# runs (the processes share the GPU and CPU).
# Usage: bash scripts/run_lora_seeds.sh [seed ...]
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
run() {  # seed
  s=$1; ck=checkpoints/moe_lora_ep6_s$s.pt
  python -u train_moe.py --init-mode lora --epochs 6 --seed "$s" --ckpt "$ck" > "logs/train_lora_ep6_s$s.log" 2>&1 \
  && python -u eval_moe.py --init-mode lora --seed "$s" --ckpt "$ck" > "logs/eval_lora_ep6_s$s.log" 2>&1 \
  && echo "== seed $s done $(date +%T)" || echo "!! seed $s failed $(date +%T)"
}
echo "start $(date +%T)"
seeds=("$@"); [ ${#seeds[@]} -eq 0 ] && seeds=(1 2 3 42)
for s in "${seeds[@]}"; do run "$s" & done
wait
grep -H "Perplexity\|Next-token" logs/eval_lora_ep6_s*.log
echo "ALL DONE $(date +%T)"
