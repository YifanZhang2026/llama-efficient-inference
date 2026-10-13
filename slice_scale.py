"""
Untrained slice-MoE vs dense FFN: output magnitude and direction, per layer.

The slice experts partition the dense FFN's hidden units (w1/w3 rows, w2
columns), so for every token
    dense_ffn(x) == sum_{i=0}^{N-1} expert_i(x)            (up to fp16 rounding)
This script measures, on Alpaca[200:300] (same tokens as eval_moe.py), how far
the actual MoE output is from that, and splits the gap into variants:

  all_w1    sum of all N experts, weight 1        -> sanity check, should == dense
  topk_w1   sum of the top-K routed experts, weight 1
  actual    sum of the top-K routed experts, weight = renormalized router prob
            (what MoEFeedForward.forward computes; checked against it)

Two settings:
  teacher-forced  every layer's FFN input is taken from the dense model, so each
                  layer's numbers are independent of the layers before it
  free-running    the converted MoE model runs on its own; compare its residual
                  stream after each layer with the dense model's

Per token: ratio = ||v|| / ||dense_ffn(x)||, cos = cosine(v, dense_ffn(x));
tables report the mean over tokens.

Usage:
    python slice_scale.py                  # seed 42, top-2 of 4
"""
import argparse
import copy
import csv
import json
import os

import torch
import torch.nn.functional as F

from llama.config import CKPT_DIR, DATA_DIR
from llama.model import ModelArgs, Llama
from llama.moe import convert_to_moe
from llama.tokenizer import Tokenizer


def build_dense(state_dict):
    args = ModelArgs()
    args.max_batch_size = 1
    args.max_seq_len = 160
    args.kv_caching = False
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(args)
    model.load_state_dict(state_dict, strict=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    return model.eval()


def segments(tokenizer, max_len=128):
    with open(os.path.join(DATA_DIR, "alpaca_500.json")) as f:
        texts = [s["text"] for s in json.load(f)[200:300]]
    for text in texts:
        tokens = tokenizer.encode(text, bos=True, eos=True)
        if len(tokens) >= 4:
            yield tokens[:max_len][:-1]  # the model inputs eval_moe uses


def ratio_cos(v, ref):
    v, ref = v.float(), ref.float()
    return v.norm(dim=-1) / ref.norm(dim=-1), F.cosine_similarity(v, ref, dim=-1)


class Acc:
    """Per-layer running sums of per-token metrics."""
    def __init__(self, n_layers):
        self.s = [dict() for _ in range(n_layers)]
        self.n = [0] * n_layers

    def add(self, layer, **metrics):
        d = self.s[layer]
        for k, t in metrics.items():
            d[k] = d.get(k, 0.0) + t.double().sum().item()

    def count(self, layer, n):
        self.n[layer] += n

    def mean(self, layer, k):
        return self.s[layer][k] / self.n[layer]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--csv", default="results/slice_scale.csv")
    a = p.parse_args()

    tokenizer = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    state_dict = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"),
                            map_location="cpu", weights_only=True, mmap=True)
    dense = build_dense(state_dict)
    torch.manual_seed(a.seed)
    moe, _ = convert_to_moe(copy.deepcopy(dense), num_experts=a.num_experts,
                            top_k=a.top_k, init_mode="slice")
    moe.eval()
    L = len(dense.layers)

    ffn_io, h_dense, h_moe = {}, {}, {}
    hooks = []
    for i, layer in enumerate(dense.layers):
        hooks.append(layer.feed_forward.register_forward_hook(
            lambda m, inp, out, i=i: ffn_io.__setitem__(i, (inp[0], out))))
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, i=i: h_dense.__setitem__(i, out)))
    for i, layer in enumerate(moe.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, i=i: h_moe.__setitem__(i, out)))

    acc = Acc(L)
    impl_err = [0.0] * L      # max |actual (recomputed) - MoEFeedForward.forward|
    all_err = [0.0] * L       # max |sum of all experts - dense FFN|
    n_tok = 0
    for seg in segments(tokenizer):
        x_ids = torch.tensor([seg], device="cuda")
        dense(x_ids, 0)
        moe(x_ids, 0)
        S = x_ids.shape[1]
        n_tok += S
        for i in range(L):
            x, y = ffn_io[i]
            x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
            ff = moe.layers[i].feed_forward
            e = torch.stack([ff._expert_forward(j, x) for j in range(ff.num_experts)])  # (N, S, dim)

            probs = F.softmax(ff.router(x).float(), dim=-1)
            w, idx = torch.topk(probs, ff.top_k, dim=-1)
            w = w / (w.sum(-1, keepdim=True) + 1e-9)
            sel = torch.zeros_like(probs).scatter_(1, idx, 1.0)          # (S, N) 0/1
            wfull = torch.zeros_like(probs).scatter_(1, idx, w)          # (S, N) weights
            all_w1 = e.float().sum(0)
            topk_w1 = torch.einsum("sn,nsd->sd", sel, e.float())
            actual = torch.einsum("sn,nsd->sd", wfull, e.float())

            impl_err[i] = max(impl_err[i], (actual - ff(x[None])[0].float()).abs().max().item())
            all_err[i] = max(all_err[i], (all_w1 - y.float()).abs().max().item())

            r_all, c_all = ratio_cos(all_w1, y)
            r_top, c_top = ratio_cos(topk_w1, y)
            r_act, c_act = ratio_cos(actual, y)
            r_exp = (e.float().norm(dim=-1) / y.float().norm(dim=-1)).mean(0)  # mean over experts
            r_h, c_h = ratio_cos(h_moe[i].reshape(S, -1), h_dense[i].reshape(S, -1))
            acc.add(i, r_all=r_all, c_all=c_all, r_top=r_top, c_top=c_top,
                    r_act=r_act, c_act=c_act, r_exp=r_exp, w_max=w[:, 0], r_h=r_h, c_h=c_h)
            acc.count(i, S)
    for h in hooks:
        h.remove()

    print(f"seed {a.seed} | top-{a.top_k} of {a.num_experts} | {n_tok} tokens (Alpaca[200:300] model inputs)")
    print("\nTeacher-forced (FFN input from the dense model); ratio = ||v||/||dense||, cos vs dense")
    print(f"{'layer':>5} | {'all_w1 r':>8} {'cos':>6} {'maxerr':>8} | {'topk_w1 r':>9} {'cos':>6} | "
          f"{'actual r':>8} {'cos':>6} | {'1 expert r':>10} | {'top1 w':>6} | {'impl err':>8}")
    rows = []
    for i in range(L):
        m = {k: acc.mean(i, k) for k in acc.s[i]}
        print(f"{i:>5} | {m['r_all']:8.4f} {m['c_all']:6.4f} {all_err[i]:8.2e} | {m['r_top']:9.4f} {m['c_top']:6.4f} | "
              f"{m['r_act']:8.4f} {m['c_act']:6.4f} | {m['r_exp']:10.4f} | {m['w_max']:6.4f} | {impl_err[i]:8.2e}")
        rows.append({"seed": a.seed, "layer": i, "n_tokens": n_tok,
                     **{k: round(v, 6) for k, v in m.items()},
                     "all_maxerr": all_err[i], "impl_maxerr": impl_err[i]})

    print("\nFree-running (residual stream after each layer, MoE model vs dense model)")
    print(f"{'layer':>5} | {'||h_moe||/||h_dense||':>21} | {'cos':>6}")
    for i in range(L):
        print(f"{i:>5} | {acc.mean(i, 'r_h'):21.4f} | {acc.mean(i, 'c_h'):6.4f}")

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
