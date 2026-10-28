"""
Every decode step's time, not just bench.py's quantiles: is a wide decode
distribution tied to the step index (KV length), to particular reps, or random?

Same model build and timing as bench.py (bench.build_model / bench.run_once:
random tokens, greedy argmax on GPU, CUDA events per step, warmup first).

Usage:
    python decode_steps.py --mode fp16 --bs 1 16 --iters 30
"""
import argparse
import csv
import datetime
import os
import statistics

import torch

import bench
from llama.config import CKPT_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="fp16", choices=["fp16", "int4"])
    p.add_argument("--bs", type=int, nargs="+", default=[1, 16])
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--csv", default="results/decode_steps.csv")
    a = p.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | mode={a.mode} "
          f"in={a.input_len} out={a.output_len} warmup={a.warmup} iters={a.iters}")
    sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    rows = []
    for bs in a.bs:
        torch.manual_seed(0)
        model = bench.build_model(sd, bs, a.input_len + a.output_len, a.mode, 128)
        tokens = torch.randint(0, model.params.vocab_size, (bs, a.input_len), device="cuda")
        for _ in range(a.warmup):
            bench.run_once(model, tokens, a.output_len)
        reps = [bench.run_once(model, tokens, a.output_len) for _ in range(a.iters)]
        for r, (pre, steps) in enumerate(reps):
            for i, ms in enumerate(steps):
                rows.append({"mode": a.mode, "bs": bs, "rep": r, "step": i, "kv_len": a.input_len + i,
                             "ms": round(ms, 4), "prefill_ms": round(pre, 4), "time": now})
        allsteps = [s for _, d in reps for s in d]
        q = statistics.quantiles(allsteps, n=10)
        print(f"\n[{a.mode}] bs={bs}: decode median {statistics.median(allsteps):.3f} ms | p10 {q[0]:.3f} | "
              f"p90 {q[-1]:.3f} | min {min(allsteps):.3f} | max {max(allsteps):.3f}")
        by_step = [statistics.median(d[i] for _, d in reps) for i in range(a.output_len - 1)]
        by_rep = [statistics.median(d) for _, d in reps]
        print("  median by step index: " + " ".join(f"{v:.2f}" for v in by_step))
        print("  median by rep:        " + " ".join(f"{v:.2f}" for v in by_rep))
        print(f"  steps < 0.9 x median: {sum(s < 0.9 * statistics.median(allsteps) for s in allsteps)} / {len(allsteps)}")
        del model, tokens
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\nappended {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
