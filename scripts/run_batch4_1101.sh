#!/bin/bash
# Batch 4 (E1e): lm-eval zero-shot tasks for the key configs, then paired comparisons.
# MMLU left out for now (near chance for a 1B model); add --mmlu to rerun with it.
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
P=/root/miniconda3/bin/python
run() { name=$1; shift; echo "== $name $(date +%T)"; $P -u eval_tasks.py --name "$name" "$@" > logs/lmeval_${name}_1101.log 2>&1 || echo "!! $name failed"; grep -E "^\s+\w+\s+acc" logs/lmeval_${name}_1101.log | head -20; }
run r1_fp16
run r2_int4_g128 --group-size 128
run e1f_clip_g128 --state-dict /root/autodl-tmp/fakequant/clip_g128.pth
run e1f_clip_g32 --state-dict /root/autodl-tmp/fakequant/clip_g32.pth
run r0_hf_fp16 --backend hf
for pair in "r1_fp16 r2_int4_g128" "r1_fp16 e1f_clip_g128" "r1_fp16 e1f_clip_g32" "r0_hf_fp16 r1_fp16"; do
  set -- $pair
  echo "== compare $1 vs $2"
  $P -u eval_tasks.py --compare $1 $2 > logs/lmeval_compare_${1}_vs_${2}_1101.log 2>&1
  tail -15 logs/lmeval_compare_${1}_vs_${2}_1101.log
done
echo "BATCH4 DONE $(date +%T)"
