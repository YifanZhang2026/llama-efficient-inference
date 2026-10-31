#!/bin/bash
# Batch 3, accuracy part (E1f-B): GPTQ and AWQ with llm-compressor (venv-quant),
# then hf_ppl.py: HF FP16 vs RTN g128 (no lm_head) vs GPTQ vs AWQ.
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
Q=/root/autodl-tmp/venv-quant/bin/python
for m in gptq awq; do
  echo "== $m $(date +%T)"
  $Q -u quant_llmc.py --method $m > logs/quant_llmc_${m}_1101.log 2>&1 || echo "!! $m failed"
  grep -E "calibration|oneshot done|saved|Error" logs/quant_llmc_${m}_1101.log | tail -4
done
echo "== hf_ppl $(date +%T)"
$Q -u hf_ppl.py --rtn --models gptq=/root/autodl-tmp/llmc/gptq_w4a16_asym awq=/root/autodl-tmp/llmc/awq_w4a16_asym > logs/hf_ppl_1101.log 2>&1
grep -E "^\s+(fp16|rtn|gptq|awq)|^\[|tokens" logs/hf_ppl_1101.log
echo "BATCH3Q DONE $(date +%T)"
