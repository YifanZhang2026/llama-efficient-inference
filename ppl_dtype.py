"""
FP16 vs BF16 for the 16-bit baseline. The released Llama-3.2-1B weights are bf16; the
repo runs everything in fp16 (bench.py / eval_ppl.py set torch.cuda.HalfTensor).

Builds our model twice from the same checkpoint, once with fp16 and once with bf16 as
the default tensor type (no model-code change), and runs eval_ppl.py's protocol with
the fp16 model as reference (PPL / acc / agree / KL).

Usage:
    python ppl_dtype.py
"""
import argparse
import csv
import datetime
import os

import torch

import eval_ppl
from llama.config import CKPT_DIR
from llama.model import ModelArgs, Llama
from llama.tokenizer import Tokenizer


def build(sd, ctx, tensor_type):
    args = ModelArgs()
    args.max_batch_size, args.max_seq_len, args.kv_caching = 1, ctx, False
    torch.set_default_tensor_type(tensor_type)
    m = Llama(args)
    m.load_state_dict(sd, strict=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    return m.eval()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--csv", default="results/ppl_dtype.csv")
    a = p.parse_args()
    sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    models = {"fp16": build(sd, a.ctx, torch.cuda.HalfTensor), "bf16": build(sd, a.ctx, torch.cuda.BFloat16Tensor)}
    print("dtypes:", {k: next(m.parameters()).dtype for k, m in models.items()})
    tok = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    ids = tok.encode(eval_ppl.load_wikitext2(), bos=True, eos=False)
    rows, now = [], datetime.datetime.now().isoformat(timespec="seconds")
    for dataset, ctx, segs in [("alpaca", 128, eval_ppl.alpaca_segments(tok)), ("wikitext2", a.ctx, eval_ppl.wikitext_segments(ids, a.ctx))]:
        r = eval_ppl.evaluate(models, segs)
        for name, q in r.items():
            extra = "" if q["agree"] is None else f" | agree {q['agree']:.2f}% | KL {q['kl']:.5f}"
            print(f"[{dataset}] ours {name}: PPL {q['ppl']:.4f} | acc {q['acc']:.2f}%{extra}", flush=True)
            rows.append({"dataset": dataset, "model": f"ours_{name}", "ppl": round(q["ppl"], 4), "acc": round(q["acc"], 3),
                         "agree_vs_fp16": "" if q["agree"] is None else round(q["agree"], 3),
                         "kl_vs_fp16": "" if q["kl"] is None else round(q["kl"], 6), "time": now})
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
