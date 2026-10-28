#!/bin/bash
# Part 3 T4 on the A100: per-layer sweep of the v4 kernel config + bs=1 end-to-end
# with (a) defaults, (b) best split=1 per shape, (c) best overall per shape.
cd "$(dirname "$0")/.."
mkdir -p logs
python -u int4_gemv_bench.py --e2e > logs/int4_gemv_bench_1029.log 2>&1
grep -v Warning logs/int4_gemv_bench_1029.log | grep -E "GPU:|N=|e2e"
tar czf artifacts_1029t4.tgz logs/int4_gemv_bench_1029.log results/int4_gemv_sweep.csv results/int4_gemv_e2e.csv
ls -la artifacts_1029t4.tgz
