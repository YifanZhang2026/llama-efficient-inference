#!/bin/bash
# T8: vLLM decode latency, FP16 vs GPTQ vs AWQ (W4A16_ASYM from quant_llmc.py).
# Needs an otherwise idle GPU. One process per model (vLLM keeps its memory pool).
cd "$(dirname "$0")/.."
V=/root/autodl-tmp/venv-vllm/bin/python
for spec in ${VLLM_MODELS:-fp16=/root/autodl-tmp/hf/Llama-3.2-1B gptq=/root/autodl-tmp/llmc/gptq_w4a16_asym awq=/root/autodl-tmp/llmc/awq_w4a16_asym}; do
  n=${spec%%=*}
  echo "== vllm $n $(date +%T)"
  $V -u vllm_bench.py --model $spec > logs/vllm_${n}_1101.log 2>&1 || echo "!! $n failed"
  grep -E "^\[vllm|Using .*(Marlin|Machete|Exllama|Kernel)|kernel" logs/vllm_${n}_1101.log | head -6
done
echo "VLLM DONE $(date +%T)"
