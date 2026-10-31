#!/bin/bash
# GPTQ weights in our INT4 format (convert_ct.py) through v4 + CUDA Graph, and their
# downstream accuracy. Starts after ppl_sd.py so the timing runs on an idle GPU.
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
P=/root/miniconda3/bin/python
until grep -q "PPLSD DONE" logs/ppl_sd_1101.log 2>/dev/null; do sleep 20; done
echo "== speed $(date +%T)"
G="--mode int4 --impl v4 --bs 1 8 16 --cuda-graph --csv results/bench_part3.csv"
$P -u bench.py $G > logs/bench_gptqv4_rtn_1101.log 2>&1
$P -u bench.py $G --int4-pack /root/autodl-tmp/fakequant/gptq_int4.pt > logs/bench_gptqv4_gptq_1101.log 2>&1
$P -u bench.py $G --int4-pack /root/autodl-tmp/fakequant/gptq_int4.pt --lm-head-fp16 > logs/bench_gptqv4_gptq_lmfp16_1101.log 2>&1
grep -h -E "loaded INT4|graph tokens|decode" logs/bench_gptqv4_*_1101.log
echo "== lm-eval $(date +%T)"
for spec in gptq_lmfp16=/root/autodl-tmp/fakequant/gptq_fake.pth gptq_lmclip=/root/autodl-tmp/fakequant/gptq_lmclip_fake.pth; do
  n=${spec%%=*}; sd=${spec#*=}
  $P -u eval_tasks.py --name $n --state-dict $sd > logs/lmeval_${n}_1101.log 2>&1 || echo "!! $n failed"
  $P -u eval_tasks.py --compare r1_fp16 $n > logs/lmeval_compare_r1_fp16_vs_${n}_1101.log 2>&1
  tail -9 logs/lmeval_compare_r1_fp16_vs_${n}_1101.log
done
echo "GPTQV4 DONE $(date +%T)"
