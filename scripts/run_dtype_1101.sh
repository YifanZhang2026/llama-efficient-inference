#!/bin/bash
# FP16 vs BF16 baseline check (2026-11-01): our model in fp16 / bf16, HF model in fp16 / bf16
# (PPL, agree, KL), and lm-eval for HF bf16 compared with the existing fp16 runs.
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
P=/root/miniconda3/bin/python
$P -u ppl_dtype.py > logs/ppl_dtype_1101.log 2>&1
grep -E "^\[|dtypes" logs/ppl_dtype_1101.log
/root/autodl-tmp/venv-quant/bin/python -u hf_ppl.py --bf16 --csv results/hf_ppl_dtype.csv > logs/hf_ppl_dtype_1101.log 2>&1
grep -E "^\s+(fp16|bf16) |^\[" logs/hf_ppl_dtype_1101.log
$P -u eval_tasks.py --name r0_hf_bf16 --backend hf --dtype bfloat16 > logs/lmeval_r0_hf_bf16_1101.log 2>&1 || echo "!! r0_hf_bf16 failed"
for ref in r0_hf_fp16 r1_fp16; do
  $P -u eval_tasks.py --compare $ref r0_hf_bf16 > logs/lmeval_compare_${ref}_vs_r0_hf_bf16_1101.log 2>&1
  echo "== $ref vs r0_hf_bf16"; tail -9 logs/lmeval_compare_${ref}_vs_r0_hf_bf16_1101.log
done
echo "DTYPE DONE $(date +%T)"
