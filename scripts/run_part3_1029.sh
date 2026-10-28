#!/bin/bash
# EXPERIMENTS.md "现阶段待做" T1-T3 on the A100 (2026-10-29): v1-v4 correctness on
# real weights, end-to-end bench, one nsys rep per version, peak-memory snapshots.
# New file names only; bench rows go to results/bench_part3.csv (untracked).
cd "$(dirname "$0")/.."
mkdir -p logs prof
echo "== correctness (real weights) $(date +%T)"
python -u - > logs/part3_check_1029.log 2>&1 <<'PY'
import os, torch
import torch.nn.functional as F
from llama.config import CKPT_DIR
from llama.quantize import QuantizedLinear, dequant_fn, int4_linear_fn
sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
for key in ["output.weight", "layers.0.attention.wq.weight", "layers.0.attention.wk.weight",
            "layers.0.feed_forward.w1.weight", "layers.0.feed_forward.w2.weight"]:
    w = sd[key].cuda()
    p, s, z = QuantizedLinear.quantize_tensor(w, 128)
    ref = dequant_fn("v1")(p, s, z, 128)
    eq = {impl: torch.equal(dequant_fn(impl)(p, s, z, 128), ref) for impl in ("v2", "v3")}
    line = f"{key:<34} {list(w.shape)} v2==v1 {eq['v2']} v3==v1 {eq['v3']}"
    for m in (1, 16):
        x = torch.randn(m, w.shape[1], dtype=torch.float16, device="cuda")
        y_ref = F.linear(x.float(), ref.float())
        y1 = F.linear(x, ref).float()
        y4 = int4_linear_fn()(x, p, s, z, 128).float()
        line += f" | M={m} max|v4-fp32ref| {(y4 - y_ref).abs().max().item():.2e} max|cuBLAS-fp32ref| {(y1 - y_ref).abs().max().item():.2e}"
    print(line, flush=True)
PY
cat logs/part3_check_1029.log | grep -v Warning | tail -6

echo "== T1 bench $(date +%T)"
python -u bench.py --mode fp16 --bs 1 8 16 --csv results/bench_part3.csv > logs/bench_part3_fp16_1029.log 2>&1
for impl in v1 v2 v3 v4; do
  python -u bench.py --mode int4 --impl $impl --bs 1 8 16 --csv results/bench_part3.csv > logs/bench_part3_int4_${impl}_1029.log 2>&1
done
grep -h "bs=" logs/bench_part3_*_1029.log

echo "== T2 nsys bs=1 $(date +%T)"
NS="nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true"
CAP="--capture-range=cudaProfilerApi --capture-range-end=stop"
$NS $CAP -o prof/p3_fp16_bs1_1029 python bench.py --mode fp16 --bs 1 --profile > logs/prof_p3_fp16_1029.log 2>&1
for impl in v1 v2 v3 v4; do
  $NS $CAP -o prof/p3_int4_${impl}_bs1_1029 python bench.py --mode int4 --impl $impl --bs 1 --profile > logs/prof_p3_int4_${impl}_1029.log 2>&1
done
for r in prof/p3_*_1029.nsys-rep; do
  nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum $r > logs/nsys_stats_$(basename $r .nsys-rep).txt 2>&1
done

echo "== T3 mem_snapshot bs=1 $(date +%T)"
for impl in v1 v2 v3 v4; do
  python -u mem_snapshot.py --mode int4 --impl $impl --bs 1 --out prof/mem_int4_${impl}_bs1_1029.pickle > logs/mem_int4_${impl}_1029.log 2>&1
  grep -i "peak" logs/mem_int4_${impl}_1029.log | head -2
done

tar czf artifacts_1029.tgz logs/*1029* results/bench_part3.csv prof/p3_*_1029.nsys-rep prof/p3_*_1029.sqlite prof/mem_int4_*_1029.pickle
ls -la artifacts_1029.tgz
echo "ALL DONE $(date +%T)"
