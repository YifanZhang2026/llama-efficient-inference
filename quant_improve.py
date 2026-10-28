"""
E1f-A: improve INT4 accuracy without calibration data, keeping the storage format
(q in [0, 15], integer zero_point, fp16 scale -> the v4 kernel runs unchanged).

Weight sets (all asymmetric, per group):
  rtn_g128 / rtn_g32     the repo's quantize_tensor (round-to-nearest over [min, max])
  clip_g128 / clip_g32   per group, shrink [min, max] by r in --ratios and keep the r
                         with the smallest reconstruction MSE of that group
Configs mix these per layer (mixed precision from E1d's sensitivity: w2 and block 1
are the most sensitive), plus FP16 for some layers. Each config reports:
  size_mb      INT4 0.5 B/weight + scale & zp 2 B each per group + 2 B/weight for
               FP16-kept layers + tok_embeddings (FP16; tied with lm_head, counted once
               only if lm_head stays FP16)
  rel_mse      ||W - W_hat||^2 / ||W||^2 over all 113 Linear
  PPL / acc / agree / KL against FP16 (eval_ppl.py protocol: Alpaca[200:300],
  WikiText-2 ctx 2048), fake quant written into fp16 Linear like int4_sensitivity.py

Usage:
    python quant_improve.py
"""
import argparse
import csv
import datetime
import os
import time

import torch
import torch.nn as nn

import eval_ppl
from llama.config import CKPT_DIR
from llama.quantize import QuantizedLinear
from llama.tokenizer import Tokenizer


def quantize_clip(w, group_size, ratios):
    """Per-group clip search. Returns (packed, scale, zp) in QuantizedLinear's format."""
    out_f, in_f = w.shape
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    gmin, gmax = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    best_err = best_scale = best_zp = best_q = None
    for r in ratios:
        mn, mx = gmin * r, gmax * r
        scale = (mx - mn) / 15
        scale = torch.where(scale == 0, torch.ones_like(scale), scale).half().float()  # as stored
        zp = torch.round(-mn / scale).clamp(0, 15)
        q = torch.round(g / scale + zp).clamp(0, 15)
        err = ((q - zp) * scale - g).pow(2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_scale, best_zp, best_q = err, scale, zp, q
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_scale = torch.where(m, scale, best_scale)
            best_zp = torch.where(m, zp, best_zp)
            best_q = torch.where(m, q, best_q)
    q = best_q.to(torch.uint8).reshape(out_f, in_f)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, best_scale.reshape(out_f, -1).half(), best_zp.reshape(out_f, -1).half()


def kind(name):
    return name.rsplit(".", 1)[-1]


def block(name):
    return int(name.split(".")[1]) if name.startswith("layers.") else None


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7])
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--only", nargs="*")
    p.add_argument("--csv", default="results/quant_improve.csv")
    a = p.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} | torch {torch.__version__} | clip ratios {a.ratios}")

    sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    ref = eval_ppl.build_model(sd, a.ctx)
    var = eval_ppl.build_model(sd, a.ctx)
    linears = {n: m for n, m in var.named_modules() if isinstance(m, nn.Linear)}
    fp16_w = {n: m.weight.data.clone() for n, m in linears.items()}

    sets, ratio_hist = {}, {}
    for sname, gs, clip in [("rtn_g128", 128, False), ("rtn_g32", 32, False), ("clip_g128", 128, True), ("clip_g32", 32, True)]:
        t0 = time.time()
        sets[sname] = {}
        for n, w in fp16_w.items():
            if clip:
                packed, scale, zp = quantize_clip(w, gs, a.ratios)
            else:
                packed, scale, zp = QuantizedLinear.quantize_tensor(w, gs)
            sets[sname][n] = QuantizedLinear.dequantize_packed(packed, scale, zp, gs)  # repo's own v1 decode
        print(f"  built {sname} in {time.time() - t0:.0f} s", flush=True)

    def rel_mse(choice):
        e = s = 0.0
        for n, w in fp16_w.items():
            wf = w.float()
            s += wf.pow(2).sum().item()
            c = choice(n)
            if c != "fp16":
                e += (sets[c][n].float() - wf).pow(2).sum().item()
        return e / s

    def size_mb(choice):
        b = 0.0
        emb = sd["tok_embeddings.weight"].numel() * 2
        for n, w in fp16_w.items():
            c = choice(n)
            if c == "fp16":
                b += 0 if n == "output" else w.numel() * 2  # tied with the embedding: stored once
            else:
                gs = int(c.split("_g")[1])
                b += w.numel() * 0.5 + (w.numel() // gs) * 4
        return (b + emb) / 2**20

    def mix(base, **over):
        """over: kind or 'block<i>' -> weight set name (or 'fp16'); block rules win over kind rules."""
        def choice(n):
            if f"block{block(n)}" in over:
                return over[f"block{block(n)}"]
            return over.get(kind(n), base)
        return choice

    configs = {
        "rtn_g128": mix("rtn_g128"),
        "rtn_g32": mix("rtn_g32"),
        "rtn_g128+w2_fp16": mix("rtn_g128", w2="fp16"),
        "rtn_g128+w2_g32": mix("rtn_g128", w2="rtn_g32"),
        "rtn_g128+block1_fp16": mix("rtn_g128", block1="fp16"),
        "rtn_g128+w2_g32+block1_fp16": mix("rtn_g128", w2="rtn_g32", block1="fp16"),
        "rtn_g128+lm_head_fp16": mix("rtn_g128", output="fp16"),
        "clip_g128": mix("clip_g128"),
        "clip_g128+w2_g32": mix("clip_g128", w2="clip_g32"),
        "clip_g128+w2_g32+block1_fp16": mix("clip_g128", w2="clip_g32", block1="fp16"),
        "clip_g32": mix("clip_g32"),
    }
    if a.only:
        configs = {k: v for k, v in configs.items() if k in a.only}

    tokenizer = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    ids = tokenizer.encode(eval_ppl.load_wikitext2(), bos=True, eos=False)
    jobs = [("alpaca", 128, lambda: eval_ppl.alpaca_segments(tokenizer)),
            ("wikitext2", a.ctx, lambda: eval_ppl.wikitext_segments(ids, a.ctx))]
    fp16_size = (sum(w.numel() for n, w in fp16_w.items() if n != "output") * 2 + sd["tok_embeddings.weight"].numel() * 2) / 2**20
    print(f"FP16 size (tied embedding counted once): {fp16_size:.1f} MB", flush=True)

    rows = []
    for cname, choice in configs.items():
        t0 = time.time()
        for n, m in linears.items():
            c = choice(n)
            m.weight.data.copy_(fp16_w[n] if c == "fp16" else sets[c][n])
        mb, mse = size_mb(choice), rel_mse(choice)
        line = []
        for dataset, ctx, segs in jobs:
            r = eval_ppl.evaluate({"fp16": ref, cname: var}, segs())
            f, q = r["fp16"], r[cname]
            line.append(f"{dataset} PPL {q['ppl']:.4f} (+{q['ppl'] - f['ppl']:.3f}) KL {q['kl']:.4f}")
            rows.append({"config": cname, "size_mb": round(mb, 1), "rel_mse": mse, "dataset": dataset, "ctx": ctx,
                         "n_tokens": q["n_tokens"], "ppl": round(q["ppl"], 4), "ppl_fp16": round(f["ppl"], 4),
                         "delta_ppl": round(q["ppl"] - f["ppl"], 4), "acc": round(q["acc"], 3),
                         "agree_vs_fp16": round(q["agree"], 3), "kl_vs_fp16": round(q["kl"], 6),
                         "ratios": " ".join(map(str, a.ratios)), "gpu": gpu, "torch": torch.__version__, "time": now})
        print(f"[{cname:<30}] {mb:7.1f} MB | rel MSE {mse * 1e3:6.3f}e-3 | " + " | ".join(line)
              + f" ({time.time() - t0:.0f} s)", flush=True)

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
