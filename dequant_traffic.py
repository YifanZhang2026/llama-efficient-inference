"""
How many HBM bytes does QuantizedLinear.dequantize_packed move per weight?

Hand count (bytes per weight, N = out * in weights; uint8 nibbles packed 2/byte):
  w_even = packed & 0x0F          read 0.5 + write 0.5          = 1
  packed >> 4                     read 0.5 + write 0.5          = 1
  (..) & 0x0F                     read 0.5 + write 0.5          = 1
  w_int4[:, 0::2] = w_even        read 0.5 + write 0.5 (stride) = 1
  w_int4[:, 1::2] = w_odd         read 0.5 + write 0.5 (stride) = 1
  w_int4.float()                  read 1   + write 4            = 5
  (w - zp)                        read 4   + write 4            = 8
  (..) * scale                    read 4   + write 4            = 8
  .half()                         read 4   + write 2            = 6
                                                          total = 32
  + F.linear reading the fp16 weight                      2  -> 34 (FP16 model: 2)

A. time every step of dequantize_packed on its own (CUDA events, warmup,
   median) and turn the hand count into an effective bandwidth per step;
B. measure the bandwidth this card actually reaches on a plain 1 GiB copy and
   predict the whole dequant time = 32 B x N / that bandwidth;
C. with --ncu, run each step exactly once between cudaProfilerStart/Stop so
   Nsight Compute can read the real DRAM bytes of every kernel:

    ncu --profile-from-start off --metrics dram__bytes_read.sum,dram__bytes_write.sum \\
        --csv python dequant_traffic.py --ncu > logs/ncu_dequant.csv

Shapes: lm_head [128256, 2048] and layer-0 wq [2048, 2048], real weights. Without
the checkpoint (or with --shape) weights are random fp16: the bytes moved depend
only on the shape. Runs with nothing but torch installed (no tiktoken needed).

Usage:
    python dequant_traffic.py                 # A + B, appends results/dequant_traffic.csv
    python dequant_traffic.py --shape big=65536x2048 --shape w1=8192x2048   # random weights
"""
import argparse
import csv
import datetime
import os
import statistics

import importlib.util

import torch
import torch.nn.functional as F

# load llama/quantize.py by path: importing the llama package pulls in the tokenizer (tiktoken)
_spec = importlib.util.spec_from_file_location(
    "llama_quantize", os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama", "quantize.py"))
_quantize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_quantize)
QuantizedLinear = _quantize.QuantizedLinear

CKPT_DIR = os.environ.get("LLAMA_CKPT_DIR", "./ckpt")  # same default as llama/config.py
SHAPES = {"lm_head": "output.weight", "wq": "layers.0.attention.wq.weight"}


def steps(packed, scale, zp, group_size):
    """dequantize_packed split into its kernels. Each entry: (name, fn, hand-count bytes)."""
    out_f, half_in = packed.shape
    in_f = half_in * 2
    n = out_f * in_f
    g = in_f // group_size
    # intermediates, so every step can be timed on its own input
    w_even = packed & 0x0F
    shr = packed >> 4
    w_odd = shr & 0x0F
    w_int4 = torch.empty(out_f, in_f, dtype=torch.uint8, device=packed.device)
    w_int4[:, 0::2] = w_even
    w_int4[:, 1::2] = w_odd
    wf = w_int4.float().reshape(out_f, g, group_size)
    zpf, sf = zp.float().unsqueeze(-1), scale.float().unsqueeze(-1)
    centered = wf - zpf
    scaled = centered * sf

    def write_even():
        w_int4[:, 0::2] = w_even

    def write_odd():
        w_int4[:, 1::2] = w_odd

    return n, [
        ("and_even  packed & 0x0F", lambda: packed & 0x0F, 1.0 * n),
        ("shr       packed >> 4", lambda: packed >> 4, 1.0 * n),
        ("and_odd   (..) & 0x0F", lambda: shr & 0x0F, 1.0 * n),
        ("write_even w_int4[:, 0::2] =", write_even, 1.0 * n),
        ("write_odd  w_int4[:, 1::2] =", write_odd, 1.0 * n),
        ("float     w_int4.float()", lambda: w_int4.float(), 5.0 * n),
        ("zp/scale  .float() (tiny)", lambda: (zp.float(), scale.float()), 0.0),
        ("sub_zp    w - zp", lambda: wf - zpf, 8.0 * n),
        ("mul_scale (..) * scale", lambda: centered * sf, 8.0 * n),
        ("half      .half()", lambda: scaled.reshape(out_f, in_f).half(), 6.0 * n),
    ]


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def copy_bandwidth(warmup, iters):
    src = torch.empty(2**30, dtype=torch.uint8, device="cuda").random_()
    dst = torch.empty_like(src)
    res = {}
    for name, fn in (("clone()", lambda: src.clone()), ("copy_()", lambda: dst.copy_(src))):
        ms = time_ms(fn, warmup, iters)
        res[name] = (ms, 2 * src.numel() / ms / 1e6)  # GB/s, read + write
    del src, dst
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--ncu", action="store_true", help="run every step once inside cudaProfilerStart/Stop")
    p.add_argument("--shape", action="append", default=[],
                   help="name=OUTxIN with random weights, repeatable (default: lm_head and wq from the checkpoint)")
    p.add_argument("--csv", default="results/dequant_traffic.csv")
    a = p.parse_args()

    ckpt = os.path.join(CKPT_DIR, "consolidated.00.pth")
    if a.shape or not os.path.exists(ckpt):
        specs = a.shape or ["lm_head=128256x2048", "wq=2048x2048"]
        shapes = {}
        for spec in specs:
            name, dims = spec.split("=")
            out_f, in_f = (int(d) for d in dims.lower().split("x"))
            shapes[name] = lambda o=out_f, i=in_f: (torch.randn(o, i, device="cuda") * 0.02).half()
        print("random weights:", ", ".join(specs))
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        shapes = {name: lambda k=key: sd[k].cuda() for name, key in SHAPES.items()}
    gpu, rows = torch.cuda.get_device_name(0), []

    if not a.ncu:
        bw = copy_bandwidth(a.warmup, a.iters)
        print(f"GPU: {gpu} | torch {torch.__version__}")
        for name, (ms, gbps) in bw.items():
            print(f"  1 GiB uint8 {name:<8} {ms:7.3f} ms -> {gbps:7.1f} GB/s (read + write)")
        ref_bw = bw["copy_()"][1]

    for shape_name, load in shapes.items():
        w = load()
        packed, scale, zp = QuantizedLinear.quantize_tensor(w, a.group_size)
        n, ops = steps(packed, scale, zp, a.group_size)
        x = torch.randn(1, 1, w.shape[1], dtype=torch.float16, device="cuda")
        w16 = QuantizedLinear.dequantize_packed(packed, scale, zp, a.group_size)
        whole = [("dequantize_packed (whole)", lambda: QuantizedLinear.dequantize_packed(packed, scale, zp, a.group_size), 32.0 * n),
                 ("F.linear fp16, 1 token", lambda: F.linear(x, w16), 2.0 * n)]

        if a.ncu:
            for _, fn, _ in ops + whole:  # warm up outside the capture
                fn()
            torch.cuda.synchronize()
            print(f"[{shape_name}] {list(w.shape)}: kernels in capture order:")
            torch.cuda.profiler.start()
            for name, fn, _ in ops:
                torch.cuda.nvtx.range_push(f"{shape_name}:{name.split()[0]}")
                fn()
                torch.cuda.nvtx.range_pop()
                print(f"  {shape_name}:{name}")
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            continue

        print(f"\n[{shape_name}] {list(w.shape)}, N = {n:,} weights, group {a.group_size}")
        print(f"  {'step':<30} {'ms':>8} {'hand-count MB':>14} {'B/weight':>9} {'GB/s':>8} {'% of copy':>9}")
        total_ms = 0.0
        for name, fn, nbytes in ops + whole:
            ms = time_ms(fn, a.warmup, a.iters)
            gbps = nbytes / ms / 1e6 if nbytes else float("nan")
            if not name.startswith(("dequantize_packed", "F.linear")):
                total_ms += ms
            print(f"  {name:<30} {ms:8.3f} {nbytes / 1e6:14.1f} {nbytes / n:9.1f} {gbps:8.1f} {100 * gbps / ref_bw:8.1f}%")
            rows.append({"shape": shape_name, "out": w.shape[0], "in": w.shape[1], "group_size": a.group_size,
                         "step": name.split()[0], "ms": round(ms, 4), "hand_bytes": int(nbytes),
                         "bytes_per_weight": nbytes / n, "eff_gbps": round(gbps, 1),
                         "copy_gbps": round(ref_bw, 1), "gpu": gpu, "torch": torch.__version__,
                         "time": datetime.datetime.now().isoformat(timespec="seconds")})
        pred = 32.0 * n / ref_bw / 1e6
        print(f"  sum of steps {total_ms:.3f} ms | predicted 32 B x N / copy bandwidth = {pred:.3f} ms")
        del w, packed, scale, zp, w16
        torch.cuda.empty_cache()

    if rows:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                wr.writeheader()
            wr.writerows(rows)
        print(f"\nappended {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
