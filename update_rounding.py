"""
How much of each SGD update survives being added to the stored weights?

Replicates train_moe.py's optimizer setup (SGD momentum 0.9, global
clip_grad_norm_ 1.0, lr_router 1e-3 / lr_expert 1e-4, max_seq_len 64, first
Alpaca samples in order) for a few steps, and at every step compares the
intended update with what actually lands in the weights:

  intended  delta = lr * (momentum * buf + grad)   computed in fp32 from the
            stored (post-clip) grad and momentum buffer, i.e. exactly what
            torch.optim.SGD adds
  ulp       spacing of the parameter's dtype at the current weight value
  lost      |delta| < ulp / 2  -> round-to-nearest gives back the old weight
  unchanged weight bit-identical after optimizer.step()

Configs (each on a freshly converted model):
  slice           router only (fp16)                  -- train_moe.py default
  slice-unfreeze  router + all expert weights (fp16)  -- --unfreeze-experts
  lora            router + LoRA A/B (fp32), control; A and B reported separately

Note: train_moe.py consumes RNG (model builds, sampling) before converting the
model it trains, so the router init here differs from the logged runs.

Usage:
    python update_rounding.py --steps 20
"""
import argparse
import csv
import gc
import json
import os

import torch
import torch.nn as nn

from llama.config import CKPT_DIR, DATA_DIR
from llama.model import ModelArgs, Llama
from llama.moe import convert_to_moe, MoEFeedForward, LoRAMoEFeedForward
from llama.tokenizer import Tokenizer

MANT_BITS = {torch.float16: 10, torch.bfloat16: 7, torch.float32: 23}
MIN_ULP = {torch.float16: 2.0 ** -24, torch.bfloat16: 2.0 ** -133, torch.float32: 2.0 ** -149}


def ulp(w):
    """Distance from w to the next representable value of w.dtype (away from zero)."""
    _, e = torch.frexp(w.float())  # |w| = m * 2^e, m in [0.5, 1)
    u = torch.exp2((e - 1 - MANT_BITS[w.dtype]).float())
    return torch.where(w == 0, torch.full_like(u, MIN_ULP[w.dtype]), u.clamp_min(MIN_ULP[w.dtype]))


def build(state_dict, init_mode, seed):
    args = ModelArgs()
    args.max_batch_size = 1
    args.max_seq_len = 160
    args.kv_caching = False
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(args)
    model.load_state_dict(state_dict, strict=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.manual_seed(seed)
    convert_to_moe(model, num_experts=4, top_k=2, init_mode=init_mode)
    return model


def param_groups(model, unfreeze, lr_router, lr_expert):
    """Same trainable set and lrs as train_moe.train_moe_router, tagged by kind."""
    for p in model.parameters():
        p.requires_grad = False
    groups = {"router": [], "expert": [], "lora_A": [], "lora_B": []}
    for layer in model.layers:
        ff = layer.feed_forward
        groups["router"] += list(ff.router.parameters())
        if isinstance(ff, LoRAMoEFeedForward):
            groups["lora_A"] += [e.lora_A.weight for e in ff.lora_experts]
            groups["lora_B"] += [e.lora_B.weight for e in ff.lora_experts]
        elif isinstance(ff, MoEFeedForward) and unfreeze:
            groups["expert"] += [p for ew in (ff.expert_w1, ff.expert_w2, ff.expert_w3)
                                 for e in ew for p in e.parameters()]
    lrs = {"router": lr_router, "expert": lr_expert, "lora_A": lr_expert, "lora_B": lr_expert}
    groups = {k: v for k, v in groups.items() if v}
    for ps in groups.values():
        for p in ps:
            p.requires_grad = True
    return groups, [{"params": ps, "lr": lrs[k], "name": k} for k, ps in groups.items()]


def run(name, state_dict, texts, tokenizer, a, unfreeze, init_mode):
    model = build(state_dict, init_mode, a.seed)
    groups, pg = param_groups(model, unfreeze, a.lr_router, a.lr_expert)
    lr = {g["name"]: g["lr"] for g in pg}
    opt = torch.optim.SGD(pg, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for ps in groups.values() for p in ps]
    model.train()
    gen = torch.Generator(device="cuda").manual_seed(0)
    rows = []

    step = 0
    for text in texts:
        tokens = tokenizer.encode(text, bos=True, eos=True)
        if len(tokens) < 4:
            continue
        step += 1
        if step > a.steps:
            break
        tokens = tokens[:64]
        x = torch.tensor([tokens[:-1]], device="cuda")
        y = torch.tensor([tokens[1:]], device="cuda")
        loss = criterion(model(x, 0).view(-1, model.params.vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()

        stats, before = {}, {}
        for k, ps in groups.items():
            s = dict(n=0, grad0=0, lost=0, nz=0, samples=[])
            for p in ps:
                u = ulp(p.data)
                s["n"] += p.numel()
                if p.grad is None:  # expert not routed to in this sample: SGD skips it
                    d = torch.zeros_like(u)
                    s["grad0"] += p.numel()
                else:
                    g = p.grad.float()
                    buf = opt.state[p].get("momentum_buffer")
                    d = lr[k] * (g if buf is None else 0.9 * buf.float() + g)
                    s["grad0"] += (p.grad == 0).sum().item()
                s["lost"] += (d.abs() < u / 2).sum().item()
                s["nz"] += (d != 0).sum().item()
                r = (d.abs() / u).flatten()
                s["samples"].append(r[torch.randint(0, r.numel(), (min(4096, r.numel()),),
                                                    device="cuda", generator=gen)])
                before[p] = (p.data.clone(), d != 0)
            stats[k] = s
        opt.step()
        for k, ps in groups.items():
            s = stats[k]
            s["unch"] = s["unch_nz"] = 0
            for p in ps:
                old, nz = before[p]
                same = p.data == old
                s["unch"] += same.sum().item()
                s["unch_nz"] += (same & nz).sum().item()
            q = torch.quantile(torch.cat(s["samples"]), torch.tensor([0.5, 0.9, 0.99], device="cuda")).tolist()
            row = {"config": name, "group": k, "step": step, "loss": round(loss.item(), 4),
                   "grad_norm_preclip": round(gnorm, 4), "n": s["n"],
                   "grad_zero_pct": 100 * s["grad0"] / s["n"],
                   "lost_pct": 100 * s["lost"] / s["n"],
                   "unchanged_pct": 100 * s["unch"] / s["n"],
                   "unchanged_of_nonzero_pct": 100 * s["unch_nz"] / max(s["nz"], 1),
                   "delta_over_ulp_p50": q[0], "delta_over_ulp_p90": q[1], "delta_over_ulp_p99": q[2]}
            rows.append(row)
        del before
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr-router", type=float, default=1e-3)
    p.add_argument("--lr-expert", type=float, default=1e-4)
    p.add_argument("--configs", nargs="+", default=["slice", "slice-unfreeze", "lora"])
    p.add_argument("--csv", default="results/update_rounding.csv")
    a = p.parse_args()

    tokenizer = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
    state_dict = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"),
                            map_location="cpu", weights_only=True, mmap=True)
    with open(os.path.join(DATA_DIR, "alpaca_500.json")) as f:
        texts = [s["text"] for s in json.load(f)][:200]

    spec = {"slice": (False, "slice"), "slice-unfreeze": (True, "slice"), "lora": (False, "lora")}
    rows = []
    for c in a.configs:
        rows += run(c, state_dict, texts, tokenizer, a, *spec[c])

    print(f"lr_router {a.lr_router}, lr_expert {a.lr_expert}, SGD momentum 0.9, clip 1.0, {a.steps} steps")
    print("lost = |lr*(0.9*buf+grad)| < ulp/2; unchanged = bit-identical after step; "
          "delta/ulp quantiles from a random subsample")
    hdr = (f"{'config':<15} {'group':<7} {'step':>4} {'loss':>7} {'gnorm':>9} | {'grad=0':>7} {'lost':>7} "
           f"{'unch':>7} {'unch|d!=0':>9} | {'d/ulp p50':>9} {'p90':>9} {'p99':>9}")
    print(hdr)
    show = {1, 2, 5, 10, a.steps}
    for r in rows:
        if r["step"] in show:
            print(f"{r['config']:<15} {r['group']:<7} {r['step']:>4} {r['loss']:>7.3f} {r['grad_norm_preclip']:>9.3f} | "
                  f"{r['grad_zero_pct']:>6.2f}% {r['lost_pct']:>6.2f}% {r['unchanged_pct']:>6.2f}% "
                  f"{r['unchanged_of_nonzero_pct']:>8.2f}% | {r['delta_over_ulp_p50']:>9.2e} "
                  f"{r['delta_over_ulp_p90']:>9.2e} {r['delta_over_ulp_p99']:>9.2e}")

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
