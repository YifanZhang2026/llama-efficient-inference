"""
Symmetric vs asymmetric per-group INT4 (round-to-nearest) on all 113 Linear layers.

Schemes (group size 128 by default; scale stored in fp16 for every scheme, as in
llama/quantize.py):
  asym   the repo's QuantizedLinear: scale = (max - min) / 15, zero_point =
         round(-min / scale) in [0, 15], q in [0, 15]  (quantize_tensor +
         dequantize_packed, bit-identical to what QuantizedLinear uses)
  sym7   zero_point = 0, scale = absmax / 7, q = clamp(round(w / scale), -8, 7)
         (-8 is never produced, so 15 levels are used)
  sym8   zero_point = 0, scale = absmax / 8, q = clamp(round(w / scale), -8, 7)
         (all 16 levels; the largest positive weights are clipped to 7 * scale)

Per layer: relative MSE = ||W - W_hat||^2 / ||W||^2 (W = the fp16 weight).
Then (unless --no-ppl) every scheme's dequantized weights are written into an
fp16 copy of the model and evaluated with eval_ppl.py's protocol (Alpaca[200:300]
and WikiText-2 ctx 2048, agreement / KL against FP16). The asym rows should
reproduce eval_ppl.py's int4_g128 numbers exactly.

Usage:
    python quant_sym.py                   # errors + PPL, appends results/quant_sym*.csv
"""
import argparse
import csv
import datetime
import os
from collections import defaultdict

import torch
import torch.nn as nn

import eval_ppl
from llama.config import CKPT_DIR
from llama.quantize import QuantizedLinear
from llama.tokenizer import Tokenizer


def dequant(w, scheme, group_size):
    """fp16 weight -> fp16 reconstruction under `scheme`."""
    if scheme == "asym":
        packed, scale, zp = QuantizedLinear.quantize_tensor(w, group_size)
        return QuantizedLinear.dequantize_packed(packed, scale, zp, group_size)
    out_f, in_f = w.shape
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    absmax = g.abs().amax(-1, keepdim=True)
    scale = (absmax / (7 if scheme == "sym7" else 8)).half().float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(g / scale).clamp(-8, 7)
    return (q * scale).reshape(out_f, in_f).half()


def kind(name):
    return name.rsplit(".", 1)[-1]  # wq / wk / wv / wo / w1 / w2 / w3 / output


def append_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--schemes", nargs="+", default=["asym", "sym7", "sym8"])
    p.add_argument("--no-ppl", action="store_true")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--csv", default="results/quant_sym.csv")
    p.add_argument("--ppl-csv", default="results/quant_sym_ppl.csv")
    a = p.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | group {a.group_size}")

    sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    max_seq_len = a.ctx
    models = {"fp16": eval_ppl.build_model(sd, max_seq_len)}
    linears = [(n, m) for n, m in models["fp16"].named_modules() if isinstance(m, nn.Linear)]
    print(f"{len(linears)} Linear layers")
    if not a.no_ppl:
        for s in a.schemes:
            models[s] = eval_ppl.build_model(sd, max_seq_len)

    # per-layer reconstruction error (and write the reconstruction into that scheme's model)
    rows = []
    err = defaultdict(float)   # (scheme, kind) -> sum sq error
    ref = defaultdict(float)   # kind -> sum sq weight
    for name, lin in linears:
        w = lin.weight.data
        wf = w.float()
        ref[kind(name)] += wf.pow(2).sum().item()
        row = {"layer": name, "kind": kind(name), "numel": w.numel(), "group_size": a.group_size}
        for s in a.schemes:
            w_hat = dequant(w, s, a.group_size)
            e = (wf - w_hat.float()).pow(2).sum().item()
            err[(s, kind(name))] += e
            row[f"rel_mse_{s}"] = e / wf.pow(2).sum().item()
            row[f"max_abs_{s}"] = (wf - w_hat.float()).abs().max().item()
            if s in models:
                models[s].get_submodule(name).weight.data.copy_(w_hat)
        row["time"] = now
        rows.append(row)
    append_csv(a.csv, rows)

    kinds = sorted(ref, key=lambda k: ["wq", "wk", "wv", "wo", "w1", "w3", "w2", "output"].index(k))
    print(f"\nrelative MSE ||W - W_hat||^2 / ||W||^2, summed over layers of each kind (x1e-3)")
    print(f"  {'kind':<8}" + "".join(f"{s:>10}" for s in a.schemes) + "".join(f"{s + '/asym':>12}" for s in a.schemes if s != "asym"))
    for k in kinds + ["all"]:
        e = {s: (sum(err[(s, kk)] for kk in kinds) / sum(ref.values())) if k == "all" else err[(s, k)] / ref[k]
             for s in a.schemes}
        ratio = "".join(f"{e[s] / e['asym']:>12.3f}" for s in a.schemes if s != "asym") if "asym" in e else ""
        print(f"  {k:<8}" + "".join(f"{1e3 * e[s]:>10.3f}" for s in a.schemes) + ratio)
    if "asym" in a.schemes:
        for s in a.schemes:
            if s != "asym":
                worse = sum(r[f"rel_mse_{s}"] > r["rel_mse_asym"] for r in rows)
                print(f"  {s}: worse than asym in {worse} / {len(rows)} layers")
    print(f"appended {len(rows)} rows to {a.csv}")

    if a.no_ppl:
        return
    tokenizer = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    ids = tokenizer.encode(eval_ppl.load_wikitext2(), bos=True, eos=False)
    jobs = [("alpaca", 128, lambda: eval_ppl.alpaca_segments(tokenizer)),
            ("wikitext2", a.ctx, lambda: eval_ppl.wikitext_segments(ids, a.ctx))]
    prow = []
    for dataset, ctx, segs in jobs:
        res = eval_ppl.evaluate(models, segs())
        print(f"\n[{dataset} ctx={ctx}]")
        for name, r in res.items():
            extra = "" if r["agree"] is None else f" | agree {r['agree']:.2f}% | KL {r['kl']:.4f}"
            print(f"  {name:<6} PPL {r['ppl']:9.4f} | acc {r['acc']:.2f}% | {r['n_tokens']} tokens{extra}")
            prow.append({"dataset": dataset, "ctx": ctx, "scheme": name, "group_size": a.group_size,
                         "n_tokens": r["n_tokens"], "ppl": round(r["ppl"], 4), "acc": round(r["acc"], 3),
                         "agree_vs_fp16": "" if r["agree"] is None else round(r["agree"], 3),
                         "kl_vs_fp16": "" if r["kl"] is None else round(r["kl"], 6),
                         "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "time": now})
    append_csv(a.ppl_csv, prow)
    print(f"\nappended {len(prow)} rows to {a.ppl_csv}")


if __name__ == "__main__":
    main()
