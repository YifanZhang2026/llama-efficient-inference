"""
T4 (EXPERIMENTS.md Part 3 / I3): tune the v4 Triton kernel on this GPU.

1. Per-layer sweep. For each Llama 3.2-1B weight shape and M in {1, 8, 16}:
   GPU time (triton.testing.do_bench) of fp16 F.linear, v1-v3 dequant + F.linear,
   v4 with its defaults, and v4 over BLOCK_N x num_warps x split-K. v4 times include
   split-K's extra kernels (zeros + .half()). Also the host-side dispatch cost per
   call (enqueue time of 200 calls, no sync): decode is launch-bound, so a config
   that wins on GPU time can still lose end to end.
2. End to end (--e2e). bs=1 decode with three v4 configurations, swapped in
   without touching llama/: (a) defaults (auto split-K), (b) best per shape among
   split=1 configs, (c) best per shape overall. Plus fp16 for reference.

Usage:
    python int4_gemv_bench.py --e2e            # sweep + end to end, ~10 min on A100
"""
import argparse
import csv
import datetime
import os
import statistics
import time

import torch
import torch.nn.functional as F
import triton.testing

import llama.quantize as quantize
from llama.quantize import QuantizedLinear, dequant_fn, int4_linear_fn

SHAPES = [(2048, 2048, "wq/wo"), (512, 2048, "wk/wv"), (8192, 2048, "w1/w3"),
          (2048, 8192, "w2"), (128256, 2048, "lm_head")]
BLOCK_N = [16, 32, 64, 128, 256]
WARPS = [2, 4, 8]
SPLITS = [1, 2, 4, 8, 16]


def gpu_us(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3


def host_us(fn, n=200):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / n * 1e6


def copy_gbps():
    src = torch.empty(2**30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    return 2 * src.numel() / gpu_us(lambda: dst.copy_(src)) / 1e3


def sweep(a, real_int4):
    rows, best = [], {}
    bw = copy_gbps()
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | copy {bw:.0f} GB/s")
    for N, K, name in SHAPES:
        if a.shapes and name not in a.shapes:
            continue
        w = (torch.randn(N, K, device="cuda") * 0.02).half()
        p, s, z = QuantizedLinear.quantize_tensor(w, 128)
        w16 = dequant_fn("v1")(p, s, z, 128)
        n_groups = K // 128
        for M in (1, 8, 16):
            x = torch.randn(M, K, dtype=torch.float16, device="cuda")
            t = {"fp16": gpu_us(lambda: F.linear(x, w16))}
            for impl in ("v1", "v2", "v3"):
                t[impl] = gpu_us(lambda: F.linear(x, dequant_fn(impl)(p, s, z, 128)))
            t["v4_default"] = gpu_us(lambda: real_int4(x, p, s, z, 128))
            host_default = host_us(lambda: real_int4(x, p, s, z, 128))
            host_fp16 = host_us(lambda: F.linear(x, w16))
            cand = []
            for bn in BLOCK_N:
                for nw in WARPS:
                    for sp in SPLITS:
                        if n_groups % sp or n_groups // sp < 1:
                            continue
                        f = lambda: real_int4(x, p, s, z, 128, block_n=bn, split_k=sp, num_warps=nw)
                        try:
                            us = gpu_us(f)
                        except Exception as e:  # e.g. out of resources for a tile size
                            continue
                        cand.append((us, bn, nw, sp))
            cand.sort()
            b_any = cand[0]
            b_one = min(c for c in cand if c[3] == 1)
            for tag, (us, bn, nw, sp) in (("any", b_any), ("split1", b_one)):
                best[(N, K, M, tag)] = dict(block_n=bn, num_warps=nw, split_k=sp)
            h_any = host_us(lambda: real_int4(x, p, s, z, 128, block_n=b_any[1], split_k=b_any[3], num_warps=b_any[2]))
            h_one = host_us(lambda: real_int4(x, p, s, z, 128, block_n=b_one[1], split_k=b_one[3], num_warps=b_one[2]))
            eff = lambda us: N * K * 0.53 / us / 1e3
            print(f"{name:>8} N={N:>6} K={K:>5} M={M:>2} | fp16 {t['fp16']:8.1f} v1 {t['v1']:8.1f} v2 {t['v2']:8.1f} "
                  f"v3 {t['v3']:8.1f} | v4 default {t['v4_default']:7.1f} best {b_any[0]:7.1f} (bn={b_any[1]} w={b_any[2]} "
                  f"split={b_any[3]}) best split=1 {b_one[0]:7.1f} (bn={b_one[1]} w={b_one[2]}) | "
                  f"v4 best {eff(b_any[0]):4.0f} GB/s ({100 * eff(b_any[0]) / bw:.0f}% copy) | host us/call: fp16 {host_fp16:.1f} "
                  f"v4 default {host_default:.1f} best {h_any:.1f} split1 {h_one:.1f}", flush=True)
            rows.append({"gpu": torch.cuda.get_device_name(0), "shape": name, "N": N, "K": K, "M": M,
                         **{f"{k}_us": round(v, 2) for k, v in t.items()},
                         "v4_best_us": round(b_any[0], 2), "best_block_n": b_any[1], "best_warps": b_any[2], "best_split": b_any[3],
                         "v4_best_split1_us": round(b_one[0], 2), "split1_block_n": b_one[1], "split1_warps": b_one[2],
                         "v4_best_gbps": round(eff(b_any[0]), 1), "copy_gbps": round(bw, 1),
                         "host_fp16_us": round(host_fp16, 2), "host_v4_default_us": round(host_default, 2),
                         "host_v4_best_us": round(h_any, 2), "host_v4_split1_us": round(h_one, 2),
                         "top5": " ".join(f"{c[0]:.1f}@{c[1]}/{c[2]}/{c[3]}" for c in cand[:5]),
                         "time": datetime.datetime.now().isoformat(timespec="seconds")})
        del w, p, s, z, w16
        torch.cuda.empty_cache()
    return rows, best


def e2e(a, real_int4, best):
    import bench
    from llama.config import CKPT_DIR
    sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)

    def make(tag):
        if tag == "default":
            return real_int4
        def f(x, p, s, z, g=128):
            M = x.numel() // x.shape[-1]
            Mb = 1 if M == 1 else (8 if M <= 8 else 16)
            cfg = best.get((p.shape[0], p.shape[1] * 2, Mb, tag))
            return real_int4(x, p, s, z, g, **cfg) if cfg else real_int4(x, p, s, z, g)
        return f

    rows = []
    for mode, tag in [("fp16", None), ("int4", "default"), ("int4", "split1"), ("int4", "any")]:
        if tag:
            quantize._int4_linear = make(tag)
        torch.manual_seed(0)
        model = bench.build_model(sd, 1, a.input_len + a.output_len, mode, 128, "v4" if mode == "int4" else "v1")
        tokens = torch.randint(0, model.params.vocab_size, (1, a.input_len), device="cuda")
        for _ in range(3):
            bench.run_once(model, tokens, a.output_len)
        reps = [bench.run_once(model, tokens, a.output_len) for _ in range(10)]
        steps = [s for _, d in reps for s in d]
        dec = statistics.median(steps)
        label = mode if not tag else f"v4-{tag}"
        print(f"[e2e bs=1] {label:<12} decode {dec:.3f} ms/step | prefill {statistics.median(p for p, _ in reps):.2f} ms", flush=True)
        rows.append({"config": label, "decode_ms": round(dec, 3), "time": datetime.datetime.now().isoformat(timespec="seconds")})
        del model, tokens
        torch.cuda.empty_cache()
    quantize._int4_linear = real_int4
    return rows


def write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e2e", action="store_true")
    p.add_argument("--shapes", nargs="*", help="subset of shape names, e.g. wk/wv lm_head")
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--csv", default="results/int4_gemv_sweep.csv")
    p.add_argument("--e2e-csv", default="results/int4_gemv_e2e.csv")
    a = p.parse_args()
    real_int4 = int4_linear_fn()
    rows, best = sweep(a, real_int4)
    write(a.csv, rows)
    if a.e2e:
        write(a.e2e_csv, e2e(a, real_int4, best))


if __name__ == "__main__":
    main()
