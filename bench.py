"""
Prefill / decode latency and peak memory for FP16, INT4 and MoE variants.

Unlike run_benchmark.py (one end-to-end generate() call, no warmup), this
drives model.forward directly:
  - random token ids, no tokenizer, no sampling -> fixed output length
  - greedy argmax on GPU between steps, no host sync inside the loop
  - CUDA events around prefill and every decode step
  - warmup reps before timing; peak memory reset after warmup
  - one fresh model per batch size, so the pre-allocated KV cache matches bs

Generating output_len tokens = 1 prefill + (output_len - 1) decode steps,
the same work as generate(max_gen_len=output_len).

Usage:
    python bench.py --mode fp16 --bs 1 8 16
    python bench.py --mode int4 --bs 1 8 16
    python bench.py --mode moe-slice --bs 1
    python bench.py --mode int4 --impl v2 --bs 1 8 16   # dequant versions, EXPERIMENTS.md Part 3

Profile one rep with Nsight Systems (only the region between
cudaProfilerStart/Stop is captured):
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        --trace=cuda,nvtx --sample=none --cpuctxsw=none -o prof/fp16_bs1 \
        python bench.py --mode fp16 --bs 1 --profile
"""
import argparse
import csv
import datetime
import gc
import os
import statistics
from contextlib import nullcontext

import torch

from llama.config import CKPT_DIR
from llama.model import ModelArgs, Llama


def build_model(state_dict, bs, max_seq_len, mode, group_size, impl="v1"):
    args = ModelArgs()
    args.max_batch_size = bs
    args.max_seq_len = max_seq_len

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(args)
    model.load_state_dict(state_dict, strict=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.eval()

    if mode == "int4":
        from llama.quantize import quantize_model, set_dequant_impl
        quantize_model(model, group_size=group_size)
        set_dequant_impl(impl)
    elif mode.startswith("moe-"):
        from llama.moe import convert_to_moe
        convert_to_moe(model, num_experts=4, top_k=2, init_mode=mode[len("moe-"):])
    return model


def nvtx(name, enabled):
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


@torch.inference_mode()
def run_once(model, tokens, output_len, use_nvtx=False):
    """One generation pass. Returns (prefill_ms, [decode_step_ms, ...])."""
    _, input_len = tokens.shape
    n_decode = output_len - 1
    events = [torch.cuda.Event(enable_timing=True) for _ in range(n_decode + 2)]

    events[0].record()
    with nvtx("prefill", use_nvtx):
        logits = model(tokens, 0)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
    events[1].record()

    for i in range(n_decode):
        with nvtx(f"decode_{i}", use_nvtx):
            logits = model(nxt, input_len + i)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
        events[i + 2].record()

    torch.cuda.synchronize()
    prefill_ms = events[0].elapsed_time(events[1])
    decode_ms = [events[i].elapsed_time(events[i + 1]) for i in range(1, n_decode + 1)]
    return prefill_ms, decode_ms


def kv_cache_mb(args, bs, max_seq_len):
    head_dim = args.dim // args.n_heads
    nbytes = 2 * args.n_layers * bs * max_seq_len * args.n_kv_heads * head_dim * 2
    return nbytes / 2**20


def bench_config(state_dict, mode, bs, a):
    max_seq_len = a.input_len + a.output_len
    torch.manual_seed(0)
    model = build_model(state_dict, bs, max_seq_len, mode, a.group_size, a.impl)
    torch.cuda.synchronize()
    static_mb = torch.cuda.memory_allocated() / 2**20
    tokens = torch.randint(0, model.params.vocab_size, (bs, a.input_len), device="cuda")

    cold = None
    for w in range(a.warmup):
        p, d = run_once(model, tokens, a.output_len)
        if w == 0:
            cold = p + sum(d)

    torch.cuda.reset_peak_memory_stats()

    if a.profile:
        torch.cuda.profiler.start()
        run_once(model, tokens, a.output_len, use_nvtx=True)
        torch.cuda.profiler.stop()
        print(f"[{mode}] bs={bs}: profiled 1 rep")
        reps = []
    else:
        reps = [run_once(model, tokens, a.output_len, use_nvtx=a.nvtx) for _ in range(a.iters)]

    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    del model, tokens
    gc.collect()
    torch.cuda.empty_cache()
    if not reps:
        return None

    prefill = [p for p, _ in reps]
    steps = [s for _, d in reps for s in d]
    e2e = [p + sum(d) for p, d in reps]
    q = statistics.quantiles(steps, n=10)
    row = {
        "mode": mode if mode != "int4" or a.impl == "v1" else f"int4-{a.impl}",
        "bs": bs,
        "input_len": a.input_len,
        "output_len": a.output_len,
        "prefill_ms": round(statistics.median(prefill), 3),
        "decode_ms": round(statistics.median(steps), 3),
        "decode_p10_ms": round(q[0], 3),
        "decode_p90_ms": round(q[-1], 3),
        "e2e_ms": round(statistics.median(e2e), 3),
        "cold_e2e_ms": round(cold, 3),
        "tok_per_s": round(bs * a.output_len / (statistics.median(e2e) / 1000), 1),
        "peak_mb": round(peak_mb, 1),
        "static_mb": round(static_mb, 1),
        "kv_cache_mb": round(kv_cache_mb(ModelArgs(), bs, max_seq_len), 1),
        "warmup": a.warmup,
        "iters": a.iters,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[{mode}] bs={bs}: prefill {row['prefill_ms']:.2f} ms | "
          f"decode {row['decode_ms']:.3f} ms/step (p10 {row['decode_p10_ms']:.3f}, p90 {row['decode_p90_ms']:.3f}) | "
          f"e2e {row['e2e_ms']:.1f} ms (cold {row['cold_e2e_ms']:.1f}) | "
          f"peak {row['peak_mb']:.0f} MB (static {row['static_mb']:.0f})")
    return row


def append_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Prefill/decode latency and peak memory benchmark.")
    p.add_argument("--mode", default="fp16", choices=["fp16", "int4", "moe-slice", "moe-lora"])
    p.add_argument("--bs", type=int, nargs="+", default=[1, 8, 16])
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--impl", default="v1", choices=["v1", "v2", "v3", "v4"],
                   help="INT4 dequant version (EXPERIMENTS.md Part 3); CSV mode becomes int4-<impl>")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--nvtx", action="store_true", help="NVTX ranges during timed reps")
    p.add_argument("--profile", action="store_true",
                   help="run one NVTX-annotated rep between cudaProfilerStart/Stop, no CSV")
    p.add_argument("--csv", default="results/bench.csv")
    return p.parse_args()


def main():
    a = parse_args()
    assert a.warmup >= 1 and a.iters >= 2 and a.output_len >= 2
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | mode={a.mode} "
          f"impl={a.impl} in={a.input_len} out={a.output_len} warmup={a.warmup} iters={a.iters}")
    state_dict = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"),
                            map_location="cpu", weights_only=True, mmap=True)
    rows = [r for bs in a.bs if (r := bench_config(state_dict, a.mode, bs, a)) is not None]
    if rows:
        append_csv(a.csv, rows)
        print(f"appended {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
