"""
eval_ppl.py's protocol on HF-format models (runs in venv-quant): FP16 reference vs
GPTQ / AWQ checkpoints from quant_llmc.py, plus an in-process RTN baseline.

  - Alpaca[200:300]: one sequence per sample, BOS + text + EOS, truncated to 128
  - WikiText-2 test: data/wikitext2_test.txt (cached by eval_ppl.py), tokenized once
    with BOS, non-overlapping --ctx windows, predict the last ctx-1 tokens of each
  - agree / KL against the HF FP16 model, per token

rtn_g128 = the repo's RTN (asymmetric, group 128, fp16 scale/zp) applied to every
Linear except lm_head. Groups run along the input dim of each row, so HF's permuted
q/k rows quantize to exactly the same values as Meta's -> comparable with E1d
`no_output`. (lm_head is tied to the embedding in HF, so it stays FP16 everywhere.)

Note: HF's Llama implements the Llama 3 RoPE scaling and our llama/model.py does not
(P3), so absolute PPLs differ from E1b; compare deltas against the FP16 row here.

Usage (venv-quant):
    /root/autodl-tmp/venv-quant/bin/python hf_ppl.py --rtn --models gptq=/root/autodl-tmp/llmc/gptq_w4a16_asym awq=/root/autodl-tmp/llmc/awq_w4a16_asym
"""
import argparse
import csv
import datetime
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def rtn_dequant(w, g=128):
    out_f, in_f = w.shape
    x = w.float().reshape(out_f, in_f // g, g)
    mn, mx = x.amin(-1), x.amax(-1)
    scale = (mx - mn) / 15
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zp = torch.round(-mn / scale).clamp(0, 15)
    q = torch.round(x / scale.unsqueeze(-1) + zp.unsqueeze(-1)).clamp(0, 15)
    deq = (q - zp.half().float().unsqueeze(-1)) * scale.half().float().unsqueeze(-1)
    return deq.reshape(out_f, in_f).half()


def load(path, quantized):
    kw = dict(dtype=torch.float16, device_map="cuda")
    if quantized:
        try:
            from transformers import CompressedTensorsConfig
            kw["quantization_config"] = CompressedTensorsConfig(run_compressed=False)  # decompress to fp16 at load
        except Exception as e:
            print("CompressedTensorsConfig unavailable, loading as is:", e)
    m = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    return m


@torch.inference_mode()
def evaluate(models, segments, ref="fp16"):
    acc = {n: dict(nll=0.0, correct=0, agree=0, kl=0.0) for n in models}
    n_tok = 0
    for seg in segments:
        x = torch.tensor([seg[:-1]], device="cuda")
        y = torch.tensor(seg[1:], device="cuda")
        n_tok += y.numel()
        ref_logp = ref_pred = None
        for name, model in models.items():
            logp = F.log_softmax(model(x).logits[0].float(), dim=-1)
            pred = logp.argmax(-1)
            a = acc[name]
            a["nll"] += -logp.gather(1, y[:, None]).sum().double().item()
            a["correct"] += (pred == y).sum().item()
            if name == ref:
                ref_logp, ref_pred = logp, pred
            else:
                a["agree"] += (pred == ref_pred).sum().item()
                a["kl"] += (ref_logp.exp() * (ref_logp - logp)).sum().double().item()
    return {n: {"n_tokens": n_tok, "ppl": math.exp(a["nll"] / n_tok), "acc": 100 * a["correct"] / n_tok,
                "agree": None if n == ref else 100 * a["agree"] / n_tok,
                "kl": None if n == ref else a["kl"] / n_tok} for n, a in acc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-path", default="/root/autodl-tmp/hf/Llama-3.2-1B")
    p.add_argument("--models", nargs="*", default=[], help="name=path of compressed-tensors checkpoints")
    p.add_argument("--rtn", action="store_true", help="add rtn_g128 (all Linear but lm_head) built in process")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--csv", default="results/hf_ppl.csv")
    a = p.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    tok = AutoTokenizer.from_pretrained(a.hf_path)
    models = {"fp16": load(a.hf_path, False)}
    if a.rtn:
        m = load(a.hf_path, False)
        n_q = 0
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and not name.endswith("lm_head"):
                mod.weight.data.copy_(rtn_dequant(mod.weight.data))
                n_q += 1
        print(f"rtn_g128: quantized {n_q} Linear (lm_head kept FP16)")
        models["rtn_g128"] = m
    for spec in a.models:
        name, path = spec.split("=", 1)
        models[name] = load(path, True)
        print(f"loaded {name} from {path}")

    with open(os.path.join(a.data_dir, "alpaca_500.json")) as f:
        texts = [s["text"] for s in json.load(f)[200:300]]
    alpaca = []
    for t in texts:
        ids = tok(t).input_ids + [tok.eos_token_id]
        if len(ids) >= 4:
            alpaca.append(ids[:128])
    with open(os.path.join(a.data_dir, "wikitext2_test.txt")) as f:
        wt = tok(f.read()).input_ids
    print(f"alpaca: {sum(len(s) - 1 for s in alpaca)} predicted tokens | wikitext2 test: {len(wt)} tokens (BOS {wt[0] == tok.bos_token_id})")
    wins = [wt[i * a.ctx:(i + 1) * a.ctx] for i in range(len(wt) // a.ctx)]

    rows = []
    for dataset, ctx, segs in [("alpaca", 128, alpaca), ("wikitext2", a.ctx, wins)]:
        res = evaluate(models, segs)
        print(f"\n[{dataset} ctx={ctx}]")
        f16 = res["fp16"]["ppl"]
        for name, r in res.items():
            extra = "" if r["agree"] is None else f" | agree {r['agree']:.2f}% | KL {r['kl']:.4f}"
            print(f"  {name:<10} PPL {r['ppl']:9.4f} (+{r['ppl'] - f16:.3f}) | acc {r['acc']:.2f}% | {r['n_tokens']} tokens{extra}")
            rows.append({"dataset": dataset, "ctx": ctx, "model": name, "n_tokens": r["n_tokens"],
                         "ppl": round(r["ppl"], 4), "delta_ppl_vs_hf_fp16": round(r["ppl"] - f16, 4),
                         "acc": round(r["acc"], 3), "agree_vs_fp16": "" if r["agree"] is None else round(r["agree"], 3),
                         "kl_vs_fp16": "" if r["kl"] is None else round(r["kl"], 6),
                         "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "time": now})
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
