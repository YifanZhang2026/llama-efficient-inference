#!/bin/bash
# LoRA MoE, 6 epochs, seeds 1-3 (seed 42 = lora_ep6 from run_sweep.sh).
# Each seed has its own checkpoint, so the three train -> eval chains run in
# parallel on one GPU. Generation speeds in these logs are NOT comparable to
# other runs (three processes share the GPU and CPU); PPL / acc are unaffected.
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
run() {  # seed
  s=$1; ck=checkpoints/moe_lora_ep6_s$s.pt
  python -u train_moe.py --init-mode lora --epochs 6 --seed "$s" --ckpt "$ck" > "logs/train_lora_ep6_s$s.log" 2>&1 \
  && python -u eval_moe.py --init-mode lora --seed "$s" --ckpt "$ck" > "logs/eval_lora_ep6_s$s.log" 2>&1 \
  && echo "== seed $s done $(date +%T)" || echo "!! seed $s failed $(date +%T)"
}
echo "start $(date +%T)"
for s in 1 2 3; do run "$s" & done
wait
grep -H "Perplexity\|Next-token" logs/eval_lora_ep6_s*.log
echo "ALL DONE $(date +%T)"
