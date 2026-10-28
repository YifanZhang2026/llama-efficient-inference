"""
Weight distribution of Llama 3.2-1B, per layer kind, and what the INT4 (g128,
asymmetric) quantizer does with it.

(a) per kind (wq / wk / wv / wo / w1 / w3 / w2 / output, plus the unquantized
    tok_embeddings): mean, std, mean/std, excess kurtosis, max |w|
(b) zero_point histogram (0..15) of the repo's quantize_tensor, and the
    distribution of per-group skew s = (max + min) / (max - min)
    (s = 0: group symmetric about 0; s > 0: shifted positive; s < 0: negative)
(c) outlier input channels: per layer, max |w| of every input column; ratio of
    the largest column to the median column, and which column indices are
    the top ones (do the same indices recur across layers?)

Usage:
    python weight_stats.py        # writes results/weight_stats*.csv
"""
import argparse
import collections
import csv
import os

import torch

from llama.config import CKPT_DIR
from llama.quantize import QuantizedLinear

KINDS = ["wq", "wk", "wv", "wo", "w1", "w3", "w2", "output", "tok_embeddings"]
SKEW_EDGES = [-1.0, -0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5, 1.0001]


def kind_of(key):
    for k in KINDS:
        if key.startswith(k + ".") or f".{k}." in key:
            return k
    return None


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--top", type=int, default=5, help="top input channels to record per layer")
    p.add_argument("--out", default="results/weight_stats")
    a = p.parse_args()
    sd = torch.load(os.path.join(CKPT_DIR, "consolidated.00.pth"), map_location="cpu", weights_only=True, mmap=True)
    keys = [k for k in sd if k.endswith(".weight") and kind_of(k) and sd[k].dim() == 2]
    print(f"{len(keys)} 2-D weights")

    mom = collections.defaultdict(lambda: torch.zeros(5, dtype=torch.float64))  # n, s1, s2, s3, s4
    amax = collections.defaultdict(float)
    zp_hist = collections.defaultdict(lambda: torch.zeros(16, dtype=torch.long))
    skew_hist = collections.defaultdict(lambda: torch.zeros(len(SKEW_EDGES) - 1, dtype=torch.long))
    skew_q = collections.defaultdict(list)
    layer_rows, chan_rows = [], []
    top_idx = collections.defaultdict(collections.Counter)

    for key in keys:
        k = kind_of(key)
        w = sd[key].cuda()
        x = w.double().flatten()
        mu = x.mean()
        m = torch.stack([torch.tensor(x.numel(), dtype=torch.float64, device=x.device),
                         x.sum(), (x ** 2).sum(), (x ** 3).sum(), (x ** 4).sum()]).cpu()
        mom[k] += m
        var = ((x - mu) ** 2).mean()
        kurt = (((x - mu) ** 4).mean() / var ** 2 - 3).item()
        amax[k] = max(amax[k], x.abs().max().item())
        row = {"key": key, "kind": k, "out": w.shape[0], "in": w.shape[1], "mean": mu.item(),
               "std": var.sqrt().item(), "excess_kurtosis": kurt, "max_abs": x.abs().max().item()}

        # (c) input-channel max |w| (for tok_embeddings: per embedding dimension)
        col = w.float().abs().amax(0)
        med = col.median().item()
        vals, idx = col.topk(a.top)
        top_idx[k].update(idx.tolist())
        row.update({"col_max_over_median": vals[0].item() / med, "col_p99_over_median": col.quantile(0.99).item() / med,
                    "top_cols": " ".join(map(str, idx.tolist()))})
        chan_rows.append({"key": key, "kind": k, **{f"top{i}": int(j) for i, j in enumerate(idx.tolist())},
                          **{f"top{i}_over_median": v / med for i, v in enumerate(vals.tolist())}})

        # (b) zero_point and per-group skew, exactly as the repo quantizer groups the weights
        if k != "tok_embeddings":
            _, _, zp = QuantizedLinear.quantize_tensor(w, a.group_size)
            zp_hist[k] += torch.bincount(zp.flatten().long(), minlength=16).cpu()
            g = w.float().reshape(w.shape[0], -1, a.group_size)
            gmax, gmin = g.amax(-1), g.amin(-1)
            s = ((gmax + gmin) / (gmax - gmin).clamp_min(1e-12)).flatten()
            skew_hist[k] += torch.bucketize(s, torch.tensor(SKEW_EDGES[1:-1], device=s.device)).bincount(
                minlength=len(SKEW_EDGES) - 1).cpu()
            skew_q[k].append(s[torch.randperm(s.numel(), device=s.device)[:20000]].cpu())
            row.update({"zp_mean": zp.float().mean().item(), "skew_median": s.median().item(),
                        "skew_abs_median": s.abs().median().item()})
        layer_rows.append(row)
        del w, x

    # (a) pooled per kind
    print(f"\n(a) per kind (pooled over layers)")
    print(f"  {'kind':<15} {'n':>12} {'mean':>11} {'std':>9} {'mean/std':>9} {'ex.kurt':>8} {'max|w|':>8} {'max|w|/std':>10}")
    kind_rows = []
    for k in KINDS:
        n, s1, s2, s3, s4 = mom[k].tolist()
        mu = s1 / n
        var = s2 / n - mu ** 2
        m4 = s4 / n - 4 * mu * s3 / n + 6 * mu ** 2 * s2 / n - 3 * mu ** 4
        kurt = m4 / var ** 2 - 3
        sd_ = var ** 0.5
        print(f"  {k:<15} {int(n):>12,} {mu:>11.2e} {sd_:>9.5f} {mu / sd_:>9.4f} {kurt:>8.2f} {amax[k]:>8.4f} {amax[k] / sd_:>10.1f}")
        kind_rows.append({"kind": k, "n": int(n), "mean": mu, "std": sd_, "mean_over_std": mu / sd_,
                          "excess_kurtosis": kurt, "max_abs": amax[k], "max_abs_over_std": amax[k] / sd_})

    print(f"\n(b) zero_point histogram (% of groups), g{a.group_size}; symmetric group -> zp 7.5")
    print(f"  {'kind':<8}" + "".join(f"{z:>5}" for z in range(16)) + f"{'mean':>7}")
    zp_rows = []
    for k in KINDS[:-1]:
        h = zp_hist[k].double()
        pct = 100 * h / h.sum()
        mean = (h * torch.arange(16)).sum().item() / h.sum().item()
        print(f"  {k:<8}" + "".join(f"{v:>5.1f}" for v in pct.tolist()) + f"{mean:>7.2f}")
        zp_rows.append({"kind": k, **{f"zp{z}_pct": round(v, 3) for z, v in enumerate(pct.tolist())}, "zp_mean": mean})

    print(f"\n(b) per-group skew s = (max+min)/(max-min), % of groups per bin; quantiles")
    labels = [f"[{SKEW_EDGES[i]:g},{SKEW_EDGES[i + 1]:g})" for i in range(len(SKEW_EDGES) - 1)]
    print(f"  {'kind':<8}" + "".join(f"{l:>13}" for l in labels) + f"{'p5':>8}{'p50':>8}{'p95':>8}")
    for k in KINDS[:-1]:
        h = skew_hist[k].double()
        pct = (100 * h / h.sum()).tolist()
        q = torch.cat(skew_q[k]).quantile(torch.tensor([0.05, 0.5, 0.95])).tolist()
        print(f"  {k:<8}" + "".join(f"{v:>13.1f}" for v in pct) + "".join(f"{v:>8.3f}" for v in q))
        for zr in zp_rows:
            if zr["kind"] == k:
                zr.update({f"skew_{l}_pct": round(v, 3) for l, v in zip(labels, pct)})
                zr.update({"skew_p5": q[0], "skew_p50": q[1], "skew_p95": q[2]})

    print(f"\n(c) input-channel max |w|: largest column / median column, over layers of each kind")
    print(f"  {'kind':<15} {'min':>7} {'median':>7} {'max':>7}   most frequent top-{a.top} column indices (count of layers)")
    for k in KINDS:
        r = [row["col_max_over_median"] for row in layer_rows if row["kind"] == k]
        r = torch.tensor(r)
        common = ", ".join(f"{i}({c})" for i, c in top_idx[k].most_common(6))
        print(f"  {k:<15} {r.min():>7.2f} {r.median():>7.2f} {r.max():>7.2f}   {common}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_csv(f"{a.out}_layers.csv", layer_rows)
    write_csv(f"{a.out}_kinds.csv", kind_rows)
    write_csv(f"{a.out}_zp_skew.csv", zp_rows)
    write_csv(f"{a.out}_channels.csv", chan_rows)


if __name__ == "__main__":
    main()
