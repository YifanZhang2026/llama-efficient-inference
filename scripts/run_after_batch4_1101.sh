#!/bin/bash
# Waits for run_batch4_1101.sh (lm-eval) to finish, then on an idle GPU:
# AWQ (it ran out of memory next to lm-eval), hf_ppl.py, and the vLLM latency runs.
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
Q=/root/autodl-tmp/venv-quant/bin/python
until grep -q "BATCH4 DONE" logs/run_batch4_1101.log 2>/dev/null; do sleep 30; done
echo "== awq $(date +%T)"
$Q -u quant_llmc.py --method awq > logs/quant_llmc_awq_1101.log 2>&1 || echo "!! awq failed"
grep -E "calibration|oneshot done|saved|Error" logs/quant_llmc_awq_1101.log | tail -4
echo "== hf_ppl $(date +%T)"
M="gptq=/root/autodl-tmp/llmc/gptq_w4a16_asym"
[ -d /root/autodl-tmp/llmc/awq_w4a16_asym ] && M="$M awq=/root/autodl-tmp/llmc/awq_w4a16_asym"
$Q -u hf_ppl.py --rtn --models $M > logs/hf_ppl_1101.log 2>&1 || echo "!! hf_ppl failed"
grep -E "^\s+(fp16|rtn_g128|gptq|awq) |^\[|tokens" logs/hf_ppl_1101.log
bash scripts/run_vllm_1101.sh
echo "AFTER4 DONE $(date +%T)"
