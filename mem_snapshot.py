"""
Find what is alive on the GPU at peak memory during one generation pass.

Records the CUDA caching allocator's history (PyTorch memory snapshot) for one
warmed-up pass (prefill + a few decode steps), then:
  - dumps the snapshot to a .pickle you can drag into https://pytorch.org/memory_viz
  - replays the trace and prints the allocations alive at the peak, each
    attributed to the innermost frame inside this repo (file:line function)

Usage:
    python mem_snapshot.py --mode int4 --bs 1
    python mem_snapshot.py --mode fp16 --bs 16
"""
import argparse
import pickle
import os
from collections import defaultdict

import torch

from bench import build_model, run_once
from llama.config import CKPT_DIR

REPO = os.path.dirname(os.path.abspath(__file__))


def site(frames):
    """Innermost frame that belongs to this repo, else the innermost frame."""
    for f in frames:
        if f["filename"].startswith(REPO) and not f["filename"].endswith("mem_snapshot.py"):
            return f"{os.path.relpath(f['filename'], REPO)}:{f['line']} {f['name']}"
    return f"{frames[0]['filename']}:{frames[0]['line']} {frames[0]['name']}" if frames else "?"


def live_at_peak(trace, baseline):
    """Replay alloc / free events; return (peak_bytes, {addr: (size, frames)} at peak)."""
    def replay(stop=None):
        live, cur, best, best_i = {}, baseline, baseline, -1
        for i, ev in enumerate(trace):
            if ev["action"] == "alloc":
                live[ev["addr"]] = (ev["size"], ev.get("frames", []))
                cur += ev["size"]
            elif ev["action"] == "free_completed" and ev["addr"] in live:
                cur -= live.pop(ev["addr"])[0]
            if cur > best:
                best, best_i = cur, i
            if stop is not None and i == stop:
                return live
        return best, best_i

    peak, peak_i = replay()
    return peak, (replay(stop=peak_i) if peak_i >= 0 else {})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--mode", default="int4", choices=["fp16", "int4", "moe-slice", "moe-lora"])
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=4, help="1 prefill + (output_len-1) decode steps")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--impl", default="v1", choices=["v1", "v2", "v3", "v4"], help="INT4 dequant version")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out", default=None, help="snapshot path (default prof/mem_<mode>_bs<bs>.pickle)")
    a = p.parse_args()
    tag = a.mode if a.mode != "int4" or a.impl == "v1" else f"int4-{a.impl}"
    out = a.out or f"prof/mem_{tag}_bs{a.bs}.pickle"

    state_dict = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"),
                            map_location="cpu", weights_only=True, mmap=True)
    model = build_model(state_dict, a.bs, a.input_len + a.output_len, a.mode, a.group_size, a.impl)
    tokens = torch.randint(0, model.params.vocab_size, (a.bs, a.input_len), device="cuda")
    run_once(model, tokens, a.output_len)  # warmup

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="python")
    run_once(model, tokens, a.output_len)
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    torch_peak = torch.cuda.max_memory_allocated()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(snap, f)

    trace = snap["device_traces"][0]
    peak, live = live_at_peak(trace, baseline)
    MB = 2**20
    print(f"[{tag}] bs={a.bs} in={a.input_len}: static {baseline / MB:.0f} MB | "
          f"peak {torch_peak / MB:.0f} MB (replayed {peak / MB:.0f}) | "
          f"transient at peak {(peak - baseline) / MB:.0f} MB in {len(live)} blocks")

    by_site = defaultdict(lambda: [0, 0])
    for size, frames in live.values():
        s = by_site[site(frames)]
        s[0] += size
        s[1] += 1
    print(f"\nalive at peak, grouped by allocation site (top {a.top}):")
    for s, (size, n) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"  {size / MB:9.1f} MB  x{n:<3d} {s}")

    print(f"\nlargest single blocks at peak:")
    for size, frames in sorted(live.values(), key=lambda v: -v[0])[:a.top]:
        print(f"  {size / MB:9.1f} MB  {site(frames)}")
    print(f"\nsnapshot: {out}  (drag into https://pytorch.org/memory_viz)")


if __name__ == "__main__":
    main()
