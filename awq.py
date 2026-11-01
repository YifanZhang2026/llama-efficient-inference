"""
AWQ (activation-aware weight quantization, Lin et al. 2024), simplified, on our model
code and our INT4 format. Written by Claude at Yifan's request (2026-11-01).

Idea: input channels with large activations matter more. For a Linear y = W x, scale
input channel c by s_c:  y = (W diag(s)) (diag(1/s) x). Exact in full precision, but
after quantizing W diag(s) the important channels get relatively finer steps.
  s = mean|x_c| ** alpha  (normalized so sqrt(max * min) = 1), alpha in a grid on [0, 1),
  chosen per group of Linears that share an input, to minimize the output error
  || Q(W s) (x / s) - W x ||^2 on calibration tokens, Q = the repo's RTN g128.
The 1/s on the input is folded into the op before it, so inference has no extra work:
  wq / wk / wv   <- attention_norm.weight
  w1 / w3        <- ffn_norm.weight
  w2             <- rows of w3 (w2's input is silu(w1 x) * (w3 x), linear in w3 x)
  wo             not scaled: its input comes from repeat_kv'd V heads (GQA), so a per-
                 channel scale cannot be folded back into wv (the AWQ reference code
                 also skips v->o when the shapes differ). lm_head: not scaled.
Blocks are processed in order; each block's FP16 output (unchanged by the fold, up to
fp16 rounding) feeds the next block, as in the AWQ reference implementation.
After all scales are folded, every Linear except lm_head is quantized with RTN g128
(<name>_rtn) or E1f's per-group clip search (<name>_clip).

Calibration: same as quant_llmc.py (128 random windows x 2048 tokens of WikiText-2
train, seed 0).

Writes, in convert_ct.py's format (so ppl_sd.py and bench.py --int4-pack work):
  <out>/<name>_{rtn,clip}_int4.pt   {Linear: (packed uint8, scale fp16, zp fp16)}
  <out>/<name>_{rtn,clip}_fake.pth  Meta-key fp16 state_dict: folded norms, dequantized
                                    INT4 Linear weights, original embedding / lm_head
Usage:
    HF_ENDPOINT=https://hf-mirror.com python awq.py --out /root/fakequant_sys
"""
import argparse
import os
import random
import time

import torch

import eval_ppl
from llama.config import CKPT_DIR, DATA_DIR
from llama.quantize import QuantizedLinear
from llama.tokenizer import Tokenizer
from quant_improve import quantize_clip

CLIP_RATIOS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


def calib_tokens(tok, n, seqlen, seed):
    path = os.path.join(DATA_DIR, "wikitext2_train.txt")
    if not os.path.exists(path):
        from datasets import load_dataset
        with open(path, "w") as f:
            f.write("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]))
    ids = tok.encode(open(path).read(), bos=True, eos=False)
    rng = random.Random(seed)
    starts = [rng.randrange(0, len(ids) - seqlen) for _ in range(n)]
    print(f"calibration: {n} x {seqlen} tokens from WikiText-2 train ({len(ids)} tokens)", flush=True)
    return torch.tensor([ids[s:s + seqlen] for s in starts])


def rtn(w, g=128):
    packed, scale, zp = QuantizedLinear.quantize_tensor(w, g)
    return QuantizedLinear.dequantize_packed(packed, scale, zp, g)


class Site:
    """Collects mean |x| per input channel and a random subset of input rows for one hook site."""

    def __init__(self, keep_per_batch, gen):
        self.sum = None
        self.count = 0
        self.rows = []
        self.keep, self.gen = keep_per_batch, gen

    def __call__(self, module, args):
        x = args[0].reshape(-1, args[0].shape[-1])
        s = x.float().abs().sum(0)
        self.sum = s if self.sum is None else self.sum + s
        self.count += x.shape[0]
        idx = torch.randint(0, x.shape[0], (self.keep,), generator=self.gen, device=x.device)
        self.rows.append(x[idx].clone())

    def absmean(self):
        return self.sum / self.count

    def x(self):
        return torch.cat(self.rows)


def search(weights, site, grid):
    """Best per-channel scale for Linears sharing one input; returns (s, alpha, loss(alpha=0), best loss)."""
    W = torch.cat([w.float() for w in weights])                 # [sum N, K]
    x = site.x().float()                                        # [T, K]
    y_ref = x @ W.t()
    am = site.absmean()
    best = (None, None, float("inf"))
    loss0 = None
    for i in range(grid):
        alpha = i / grid
        s = am.pow(alpha).clamp_min(1e-4)
        s = s / (s.max() * s.min()).sqrt()
        wq = torch.cat([rtn((w.float() * s).half()) for w in weights]).float()
        loss = ((x / s) @ wq.t() - y_ref).pow(2).mean().item()
        if i == 0:
            loss0 = loss
        if loss < best[2]:
            best = (s, alpha, loss)
    return best[0], best[1], loss0, best[2]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="awq_ours")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--loss-tokens", type=int, default=16384, help="input rows kept per site for the scale search")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default="/root/fakequant_sys")
    a = p.parse_args()
    t_start = time.time()
    ck = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    model = eval_ppl.build_model(ck, a.seqlen)
    tok = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    ids = calib_tokens(tok, a.nsamples, a.seqlen, a.seed).cuda()
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    keep = max(1, a.loss_tokens // (a.nsamples // a.batch))

    freqs = model.freqs_cis.to("cuda")[:a.seqlen]
    mask = torch.triu(torch.full((a.seqlen, a.seqlen), float("-inf"), device="cuda"), diagonal=1)
    h = [model.tok_embeddings(ids[b:b + a.batch]) for b in range(0, a.nsamples, a.batch)]

    for i, layer in enumerate(model.layers):
        att, ff = layer.attention, layer.feed_forward
        sites = {k: Site(keep, gen) for k in ("qkv", "gateup", "down")}
        hooks = [att.wq.register_forward_pre_hook(sites["qkv"]),
                 ff.w1.register_forward_pre_hook(sites["gateup"]),
                 ff.w2.register_forward_pre_hook(sites["down"])]
        h = [layer(x, 0, freqs, mask) for x in h]               # FP16 block output = next block's input
        for hk in hooks:
            hk.remove()
        msg = []
        # wq / wk / wv: input = attention_norm(x)  ->  fold 1/s into attention_norm.weight
        s, al, l0, l1 = search([att.wq.weight, att.wk.weight, att.wv.weight], sites["qkv"], a.grid)
        layer.attention_norm.weight.data = (layer.attention_norm.weight.float() / s).half()
        for w in (att.wq, att.wk, att.wv):
            w.weight.data = (w.weight.float() * s).half()
        msg.append(f"qkv a={al:.2f} {l1 / l0:.3f}")
        # w1 / w3: input = ffn_norm(h)  ->  fold into ffn_norm.weight
        s, al, l0, l1 = search([ff.w1.weight, ff.w3.weight], sites["gateup"], a.grid)
        layer.ffn_norm.weight.data = (layer.ffn_norm.weight.float() / s).half()
        for w in (ff.w1, ff.w3):
            w.weight.data = (w.weight.float() * s).half()
        msg.append(f"gate/up a={al:.2f} {l1 / l0:.3f}")
        # w2: input = silu(w1 x) * (w3 x)  ->  fold 1/s into the rows of w3
        s, al, l0, l1 = search([ff.w2.weight], sites["down"], a.grid)
        ff.w3.weight.data = (ff.w3.weight.float() / s[:, None]).half()
        ff.w2.weight.data = (ff.w2.weight.float() * s).half()
        msg.append(f"down a={al:.2f} {l1 / l0:.3f}")
        print(f"block {i:2d}: " + " | ".join(msg) + "   (best output MSE / MSE at alpha=0)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "cache" not in k}
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != "output"]
    for variant in ("rtn", "clip"):
        int4, fake = {}, dict(base)
        for n in linears:
            w = model.get_submodule(n).weight.data
            if variant == "rtn":
                packed, scale, zp = QuantizedLinear.quantize_tensor(w, 128)
            else:
                packed, scale, zp = quantize_clip(w, 128, CLIP_RATIOS)
            int4[n] = (packed.cpu(), scale.cpu(), zp.cpu())
            fake[n + ".weight"] = QuantizedLinear.dequantize_packed(packed, scale, zp, 128).cpu()
        name = f"{a.name}_{variant}"
        torch.save({"int4": int4, "group_size": 128, "source": "awq.py"}, os.path.join(a.out, f"{name}_int4.pt"))
        torch.save(fake, os.path.join(a.out, f"{name}_fake.pth"))
        print(f"saved {a.out}/{name}_int4.pt and {name}_fake.pth", flush=True)
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
