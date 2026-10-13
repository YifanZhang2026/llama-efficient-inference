"""
Perplexity / next-token accuracy of FP16 vs INT4 (weight-only RTN, per-group).

Two eval sets:
  alpaca    Alpaca[200:300], same protocol as eval_moe.evaluate_perplexity:
            one sequence per sample, encode(bos=True, eos=True), truncated to
            128 tokens, predict tokens[1:] from tokens[:-1].
  wikitext2 WikiText-2 raw test, GPTQ/AWQ protocol: "\\n\\n".join(test["text"]),
            tokenized once (BOS at the very start only), cut into
            non-overlapping windows of --ctx tokens, tail dropped. Each window
            predicts its last ctx-1 tokens; PPL = exp(total NLL / predicted
            tokens). (GPTQ divides by ctx instead of ctx-1, <0.1% difference.)

All models are built once and evaluated on the same segments in one pass, so
besides PPL / acc each INT4 row also reports, against the FP16 reference:
  agree  fraction of positions where argmax(INT4) == argmax(FP16)
  kl     mean KL(P_fp16 || P_int4) per token (nats)

Note: llama/model.py does not implement Llama 3.1+ RoPE scaling (EXPERIMENTS.md
P3), so absolute WikiText PPL may differ slightly from HF reference numbers;
FP16 and INT4 share the same model code, so the FP16 -> INT4 delta is unaffected.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python eval_ppl.py                 # fp16 + int4 g128, both sets
    python eval_ppl.py --group-sizes 32 64 128 --ctx 2048 1024
"""
import argparse
import csv
import datetime
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from llama.config import CKPT_DIR, DATA_DIR
from llama.model import ModelArgs, Llama
from llama.quantize import quantize_model
from llama.tokenizer import Tokenizer

WIKITEXT_CACHE = os.path.join(DATA_DIR, "wikitext2_test.txt")


def build_model(state_dict, max_seq_len, group_size=None):
    args = ModelArgs()
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    args.kv_caching = False
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(args)
    model.load_state_dict(state_dict, strict=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.eval()
    if group_size is not None:
        quantize_model(model, group_size=group_size)
    return model


def model_mb(model):
    n = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    return n / 2**20


def load_wikitext2():
    if not os.path.exists(WIKITEXT_CACHE):
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        with open(WIKITEXT_CACHE, "w") as f:
            f.write("\n\n".join(ds["text"]))
    with open(WIKITEXT_CACHE) as f:
        return f.read()


def alpaca_segments(tokenizer, max_len=128):
    with open(os.path.join(DATA_DIR, "alpaca_500.json")) as f:
        texts = [s["text"] for s in json.load(f)[200:300]]
    for text in texts:
        tokens = tokenizer.encode(text, bos=True, eos=True)
        if len(tokens) < 4:
            continue
        yield tokens[:max_len]


def wikitext_segments(ids, ctx):
    for i in range(len(ids) // ctx):
        yield ids[i * ctx:(i + 1) * ctx]


@torch.inference_mode()
def evaluate(models, segments, ref="fp16"):
    """models: {name: model}. Returns {name: stats dict}."""
    acc = {n: dict(nll=0.0, correct=0, agree=0, kl=0.0) for n in models}
    n_tok = 0
    for seg in segments:
        x = torch.tensor([seg[:-1]], device="cuda")
        y = torch.tensor(seg[1:], device="cuda")
        n_tok += y.numel()
        ref_logp = ref_pred = None
        for name, model in models.items():  # ref is inserted first
            logp = F.log_softmax(model(x, 0)[0], dim=-1)  # fp32 (S, vocab)
            pred = logp.argmax(-1)
            a = acc[name]
            a["nll"] += -logp.gather(1, y[:, None]).sum().double().item()
            a["correct"] += (pred == y).sum().item()
            if name == ref:
                ref_logp, ref_pred = logp, pred
            else:
                a["agree"] += (pred == ref_pred).sum().item()
                a["kl"] += (ref_logp.exp() * (ref_logp - logp)).sum().double().item()
            del logp
    out = {}
    for name, a in acc.items():
        out[name] = {
            "n_tokens": n_tok,
            "nll": a["nll"] / n_tok,
            "ppl": math.exp(a["nll"] / n_tok),
            "acc": 100 * a["correct"] / n_tok,
            "agree": None if name == ref else 100 * a["agree"] / n_tok,
            "kl": None if name == ref else a["kl"] / n_tok,
        }
    return out


def append_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="FP16 vs INT4 perplexity / next-token accuracy.")
    p.add_argument("--datasets", nargs="+", default=["alpaca", "wikitext2"], choices=["alpaca", "wikitext2"])
    p.add_argument("--group-sizes", type=int, nargs="+", default=[128], help="INT4 group sizes to evaluate")
    p.add_argument("--ctx", type=int, nargs="+", default=[2048], help="WikiText-2 window lengths")
    p.add_argument("--csv", default="results/ppl.csv")
    return p.parse_args()


def main():
    a = parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    tokenizer = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    state_dict = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"),
                            map_location="cpu", weights_only=True, mmap=True)
    max_seq_len = max(a.ctx) if "wikitext2" in a.datasets else 160

    models = {"fp16": build_model(state_dict, max_seq_len)}
    for g in a.group_sizes:
        models[f"int4_g{g}"] = build_model(state_dict, max_seq_len, group_size=g)
    for name, m in models.items():
        print(f"  {name}: params + buffers = {model_mb(m):.1f} MB")

    jobs = []
    if "alpaca" in a.datasets:
        jobs.append(("alpaca", 128, lambda: alpaca_segments(tokenizer)))
    if "wikitext2" in a.datasets:
        ids = tokenizer.encode(load_wikitext2(), bos=True, eos=False)
        print(f"  wikitext2 test: {len(ids)} tokens")
        for ctx in a.ctx:
            jobs.append(("wikitext2", ctx, lambda ctx=ctx: wikitext_segments(ids, ctx)))

    rows = []
    for dataset, ctx, segs in jobs:
        t0 = time.time()
        res = evaluate(models, segs())
        print(f"\n[{dataset} ctx={ctx}] ({time.time() - t0:.0f} s)")
        for name, r in res.items():
            extra = "" if r["agree"] is None else f" | agree {r['agree']:.2f}% | KL {r['kl']:.4f}"
            print(f"  {name:<10} PPL {r['ppl']:9.4f} | acc {r['acc']:.2f}% | {r['n_tokens']} tokens{extra}")
            rows.append({
                "dataset": dataset, "ctx": ctx, "model": name,
                "model_mb": round(model_mb(models[name]), 1),
                "n_tokens": r["n_tokens"], "ppl": round(r["ppl"], 4), "nll": round(r["nll"], 6),
                "acc": round(r["acc"], 3),
                "agree_vs_fp16": "" if r["agree"] is None else round(r["agree"], 3),
                "kl_vs_fp16": "" if r["kl"] is None else round(r["kl"], 6),
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
            })
    append_csv(a.csv, rows)
    print(f"\nappended {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
