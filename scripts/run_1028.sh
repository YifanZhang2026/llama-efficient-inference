#!/bin/bash
# EXP_HANDOFF items 8 (weight distribution), 9 (FP16 decode bs=16 vs bs=1) and
# 11 (which layers the INT4 accuracy loss comes from), 2026-10-28. Sequential;
# quick steps first, the long sensitivity sweep last. New log names only.
cd "$(dirname "$0")/.."
mkdir -p logs prof
echo "== weight_stats $(date +%T)"
python -u weight_stats.py > logs/weight_stats_1028.log 2>&1

echo "== decode_steps (item 9) $(date +%T)"
python -u decode_steps.py --mode fp16 --bs 1 16 --iters 30 > logs/decode_steps_1028.log 2>&1

echo "== nsys fp16 bs=1 / bs=16 (item 9) $(date +%T)"
NS="nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true"
CAP="--capture-range=cudaProfilerApi --capture-range-end=stop"
for bs in 1 16; do
  $NS $CAP -o prof/fp16_bs${bs}_1028 python bench.py --mode fp16 --bs $bs --profile > logs/prof_fp16_bs${bs}_1028.log 2>&1
  nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum prof/fp16_bs${bs}_1028.nsys-rep \
    > logs/nsys_stats_fp16_bs${bs}_1028.txt 2>&1
done

echo "== int4_sensitivity (item 11, incl. per-layer) $(date +%T)"
python -u int4_sensitivity.py --per-layer > logs/int4_sensitivity_1028.log 2>&1

tar czf artifacts_1028.tgz logs/*1028* results/*.csv results/int4_generations.md \
  prof/fp16_bs*_1028.nsys-rep prof/fp16_bs*_1028.sqlite
ls -la artifacts_1028.tgz
echo "ALL DONE $(date +%T)"
