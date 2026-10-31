"""
eval_ppl.py's protocol (our model code, Alpaca[200:300] + WikiText-2 ctx 2048, agree /
KL vs FP16) for converted GPTQ / AWQ weights (convert_ct.py), each with two lm_head
choices:
  <name>_lmfp16   lm_head FP16 (as in the GPTQ / AWQ checkpoint; tied with the embedding)
  <name>_lmclip   lm_head quantized with E1f's clip_g128 (same as the rest of our INT4 path)
Also checks that the converted INT4 buffers decode, through the repo's own
QuantizedLinear.dequantize_packed, to exactly the fake-quant weights (so the v1-v4
kernels run these weights unchanged). The _lmclip state_dicts are saved next to the
inputs for eval_tasks.py --state-dict (only for --save-lmclip names; 3 GB each).

Usage (main env):
    python ppl_sd.py --names gptq awq
"""
import argparse
import csv
import datetime
import os

import torch

import eval_ppl
from llama.config import CKPT_DIR
from llama.quantize import QuantizedLinear
from llama.tokenizer import Tokenizer
from quant_improve import quantize_clip


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", default=["gptq", "awq"])
    p.add_argument("--dir", default="/root/autodl-tmp/fakequant")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--save-lmclip", nargs="*", default=["gptq"], help="names whose _lmclip state_dict to save (3 GB each)")
    p.add_argument("--csv", default="results/ppl_sd.csv")
    a = p.parse_args()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    gpu = torch.cuda.get_device_name(0)
    ck = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    ref = eval_ppl.build_model(ck, a.ctx)
    var = eval_ppl.build_model(ck, a.ctx)
    tok = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    ids = tok.encode(eval_ppl.load_wikitext2(), bos=True, eos=False)
    jobs = [("alpaca", 128, lambda: eval_ppl.alpaca_segments(tok)),
            ("wikitext2", a.ctx, lambda: eval_ppl.wikitext_segments(ids, a.ctx))]
    clip_out = None
    rows = []
    for name in a.names:
        fake = torch.load(os.path.join(a.dir, f"{name}_fake.pth"), map_location="cpu", weights_only=True)
        int4 = torch.load(os.path.join(a.dir, f"{name}_int4.pt"), map_location="cpu", weights_only=True)["int4"]
        worst = 0.0
        for ln, (packed, scale, zp) in int4.items():
            w = QuantizedLinear.dequantize_packed(packed.cuda(), scale.cuda(), zp.cuda(), 128)
            worst = max(worst, (w.float() - fake[ln + ".weight"].cuda().float()).abs().max().item())
        print(f"[{name}] {len(int4)} INT4 Linear: max |QuantizedLinear.dequantize_packed - fake| = {worst:.3e}", flush=True)
        if clip_out is None:  # clip_g128 lm_head, same for every name (lm_head is the original FP16 weight)
            packed, scale, zp = quantize_clip(ck["output.weight"].cuda(), 128, [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7])
            clip_out = QuantizedLinear.dequantize_packed(packed, scale, zp, 128).cpu()
        for lm in ("lmfp16", "lmclip"):
            sd = dict(fake)
            if lm == "lmclip":
                sd["output.weight"] = clip_out
                if name in a.save_lmclip:
                    torch.save(sd, os.path.join(a.dir, f"{name}_lmclip_fake.pth"))
            var.load_state_dict(sd, strict=False)
            cname = f"{name}_{lm}"
            line = []
            for dataset, ctx, segs in jobs:
                r = eval_ppl.evaluate({"fp16": ref, cname: var}, segs())
                f16, q = r["fp16"], r[cname]
                line.append(f"{dataset} PPL {q['ppl']:.4f} (+{q['ppl'] - f16['ppl']:.3f}) KL {q['kl']:.4f}")
                rows.append({"config": cname, "dataset": dataset, "ctx": ctx, "n_tokens": q["n_tokens"],
                             "ppl": round(q["ppl"], 4), "ppl_fp16": round(f16["ppl"], 4),
                             "delta_ppl": round(q["ppl"] - f16["ppl"], 4), "acc": round(q["acc"], 3),
                             "agree_vs_fp16": round(q["agree"], 3), "kl_vs_fp16": round(q["kl"], 6),
                             "gpu": gpu, "torch": torch.__version__, "time": now})
            print(f"[{cname:<12}] " + " | ".join(line), flush=True)
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
