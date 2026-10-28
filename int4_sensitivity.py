"""
Which layers does the INT4 (g128, asymmetric RTN) accuracy loss come from?

Every config quantizes a subset of the 113 Linear layers and keeps the rest in
FP16 ("fake quant": the dequantized INT4 weight is written into an fp16
nn.Linear, numerically identical to QuantizedLinear.forward with the v1 dequant).
Each config is evaluated with eval_ppl.py's protocol (Alpaca[200:300] and
WikiText-2 ctx 2048, PPL / acc / agreement / KL against FP16).

  all              all 113 (should reproduce E1b int4_g128: 11.0276 / 12.2879)
  no_output        all except lm_head (output)                       (a)
  only_<kind>      only wq / wk / wv / wo / w1 / w3 / w2 / output     (b)
  only_layer<i>    only the 7 Linear of transformer block i           (c, --per-layer)

Then greedy generations (no sampling) of FP16 and INT4 (all) side by side.

Usage:
    python int4_sensitivity.py                 # all, (a), (b), generations
    python int4_sensitivity.py --per-layer     # also (c)
"""
import argparse
import csv
import datetime
import json
import os
import time

import torch
import torch.nn as nn

import eval_ppl
from llama.config import CKPT_DIR, DATA_DIR
from llama.quantize import QuantizedLinear
from llama.tokenizer import Tokenizer

KINDS = ["wq", "wk", "wv", "wo", "w1", "w3", "w2", "output"]


def kind(name):
    return name.rsplit(".", 1)[-1]


@torch.inference_mode()
def greedy(model, tokenizer, prompt, n_new):
    ids = tokenizer.encode(prompt, bos=True, eos=False)
    start = len(ids)
    for _ in range(n_new):
        nxt = model(torch.tensor([ids], device="cuda"), 0)[0, -1].argmax().item()
        if nxt in tokenizer.stop_tokens:
            break
        ids.append(nxt)
    return tokenizer.decode(ids[start:])


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--per-layer", action="store_true", help="also quantize one transformer block at a time")
    p.add_argument("--only", nargs="*", help="run just these config names")
    p.add_argument("--gen-prompts", type=int, default=6)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--csv", default="results/int4_sensitivity.csv")
    p.add_argument("--gen-out", default="results/int4_generations.md")
    a = p.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} | torch {torch.__version__} | g{a.group_size}")

    sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    ref = eval_ppl.build_model(sd, a.ctx)
    var = eval_ppl.build_model(sd, a.ctx)          # weights swapped in and out per config
    linears = {n: m for n, m in var.named_modules() if isinstance(m, nn.Linear)}
    fp16_w, q_w = {}, {}
    for n, m in linears.items():
        fp16_w[n] = m.weight.data.clone()
        packed, scale, zp = QuantizedLinear.quantize_tensor(m.weight.data, a.group_size)
        q_w[n] = QuantizedLinear.dequantize_packed(packed, scale, zp, a.group_size)
    print(f"{len(linears)} Linear layers quantized once; fp16 and int4 copies kept on GPU")

    configs = {"all": list(linears), "no_output": [n for n in linears if kind(n) != "output"]}
    configs.update({f"only_{k}": [n for n in linears if kind(n) == k] for k in KINDS})
    if a.per_layer:
        configs.update({f"only_layer{i}": [n for n in linears if n.startswith(f"layers.{i}.")] for i in range(16)})
    if a.only:
        configs = {k: v for k, v in configs.items() if k in a.only}

    tokenizer = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    ids = tokenizer.encode(eval_ppl.load_wikitext2(), bos=True, eos=False)
    jobs = [("alpaca", 128, lambda: eval_ppl.alpaca_segments(tokenizer)),
            ("wikitext2", a.ctx, lambda: eval_ppl.wikitext_segments(ids, a.ctx))]

    rows = []
    for cname, names in configs.items():
        t0 = time.time()
        for n in linears:
            linears[n].weight.data.copy_(q_w[n] if n in names else fp16_w[n])
        qparams = sum(fp16_w[n].numel() for n in names)
        line = []
        for dataset, ctx, segs in jobs:
            r = eval_ppl.evaluate({"fp16": ref, cname: var}, segs())
            f, q = r["fp16"], r[cname]
            line.append(f"{dataset} PPL {q['ppl']:.4f} (+{q['ppl'] - f['ppl']:.3f}) acc {q['acc']:.2f}% KL {q['kl']:.4f}")
            rows.append({"config": cname, "n_linear": len(names), "quantized_params": qparams,
                         "dataset": dataset, "ctx": ctx, "n_tokens": q["n_tokens"],
                         "ppl": round(q["ppl"], 4), "ppl_fp16": round(f["ppl"], 4),
                         "delta_ppl": round(q["ppl"] - f["ppl"], 4), "acc": round(q["acc"], 3),
                         "acc_fp16": round(f["acc"], 3), "agree_vs_fp16": round(q["agree"], 3),
                         "kl_vs_fp16": round(q["kl"], 6), "group_size": a.group_size,
                         "gpu": gpu, "torch": torch.__version__, "time": now})
        print(f"[{cname:<13}] {len(names):>3} Linear, {qparams / 1e6:7.1f} M params | " + " | ".join(line)
              + f" ({time.time() - t0:.0f} s)", flush=True)

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {a.csv}")

    if a.gen_prompts:
        for n in linears:
            linears[n].weight.data.copy_(q_w[n])
        with open(os.path.join(DATA_DIR, "mt_bench_prompts.json")) as f:
            prompts = json.load(f)[:a.gen_prompts]
        out = [f"# FP16 vs INT4 g{a.group_size} greedy generations\n",
               f"{now} · {gpu} · greedy (argmax), {a.gen_tokens} new tokens max, no KV cache · `int4_sensitivity.py`\n"]
        for i, prompt in enumerate(prompts):
            g16, g4 = greedy(ref, tokenizer, prompt, a.gen_tokens), greedy(var, tokenizer, prompt, a.gen_tokens)
            out += [f"\n## {i + 1}. {prompt}\n", f"\n**FP16**\n\n> {g16.strip().replace(chr(10), ' ')}\n",
                    f"\n**INT4 g{a.group_size}**\n\n> {g4.strip().replace(chr(10), ' ')}\n"]
            print(f"\n[{i + 1}] {prompt}\n  FP16: {g16.strip()[:200]!r}\n  INT4: {g4.strip()[:200]!r}")
        with open(a.gen_out, "w") as f:
            f.write("".join(out))
        print(f"\nwrote {a.gen_out}")


if __name__ == "__main__":
    main()
