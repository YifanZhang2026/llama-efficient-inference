"""
FP16 GEMM throughput over time (default 4096^3, 30 s), one line per ~1 s window
with a wall-clock timestamp, to line up with nvidia-smi dmon / query logs.

Usage:
    python gemm_smoke.py --n 4096 --seconds 30
"""
import argparse
import datetime
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--seconds", type=float, default=30)
    a = p.parse_args()
    n = a.n
    x = torch.randn(n, n, dtype=torch.float16, device="cuda")
    y = torch.randn(n, n, dtype=torch.float16, device="cuda")
    out = torch.empty(n, n, dtype=torch.float16, device="cuda")
    flop = 2 * n ** 3
    for _ in range(20):
        torch.matmul(x, y, out=out)
    torch.cuda.synchronize()
    # matmuls per ~1 s window, from a short probe
    a0, b0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a0.record()
    for _ in range(50):
        torch.matmul(x, y, out=out)
    b0.record()
    torch.cuda.synchronize()
    per_window = max(1, int(50 * 1000 / a0.elapsed_time(b0)))
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {n}^3 fp16, {per_window} matmuls per window")
    print("wall_time,window_ms,tflops")
    t_end = time.time() + a.seconds
    best = 0.0
    while time.time() < t_end:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(per_window):
            torch.matmul(x, y, out=out)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e)
        tf = flop * per_window / ms / 1e9
        best = max(best, tf)
        print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]},{ms:.1f},{tf:.1f}", flush=True)
    print(f"best window {best:.1f} TFLOPS")


if __name__ == "__main__":
    main()
