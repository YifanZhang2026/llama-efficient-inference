#!/bin/bash
# Part 3 T6 (CUDA Graph decode) and T7 (tinygemm) on the A100, 2026-10-29.
cd "$(dirname "$0")/.."
mkdir -p logs prof
echo "== T6 bench --cuda-graph $(date +%T)"
python -u bench.py --mode fp16 --bs 1 8 16 --cuda-graph --csv results/bench_part3.csv > logs/bench_graph_fp16_1029.log 2>&1
for impl in v2 v3 v4; do
  python -u bench.py --mode int4 --impl $impl --bs 1 8 16 --cuda-graph --csv results/bench_part3.csv > logs/bench_graph_int4_${impl}_1029.log 2>&1
done
grep -h "bs=" logs/bench_graph_*_1029.log
echo "== T6 nsys (graph, bs=1) $(date +%T)"
NS="nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true --cuda-graph-trace=node"
CAP="--capture-range=cudaProfilerApi --capture-range-end=stop"
$NS $CAP -o prof/p3g_fp16_bs1_1029 python bench.py --mode fp16 --bs 1 --cuda-graph --profile > logs/prof_p3g_fp16_1029.log 2>&1
$NS $CAP -o prof/p3g_int4_v4_bs1_1029 python bench.py --mode int4 --impl v4 --bs 1 --cuda-graph --profile > logs/prof_p3g_int4_v4_1029.log 2>&1
for r in prof/p3g_*_1029.nsys-rep; do
  nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum $r > logs/nsys_stats_$(basename $r .nsys-rep).txt 2>&1
done
echo "== T6 v4 configs under CUDA Graph (configs from the T4 sweep) $(date +%T)"
python -u int4_gemv_bench.py --e2e --cuda-graph --from-csv results/int4_gemv_sweep.csv --e2e-csv results/int4_gemv_e2e.csv > logs/int4_gemv_graph_1029.log 2>&1
grep -v Warning logs/int4_gemv_graph_1029.log | grep -E "loaded|graph tokens|e2e"
echo "== T7 tinygemm $(date +%T)"
python -u tinygemm_bench.py --e2e > logs/tinygemm_1029.log 2>&1
grep -v Warning logs/tinygemm_1029.log | grep -E "GPU:|N=|e2e"
tar czf artifacts_1029t67.tgz logs/bench_graph_*_1029.log logs/prof_p3g_*_1029.log logs/nsys_stats_p3g_*_1029.txt \
  logs/tinygemm_1029.log logs/int4_gemv_graph_1029.log results/int4_gemv_e2e.csv results/bench_part3.csv results/tinygemm*.csv prof/p3g_*_1029.nsys-rep prof/p3g_*_1029.sqlite
ls -la artifacts_1029t67.tgz
echo "ALL DONE $(date +%T)"
