"""
E1e: downstream task accuracy with lm-evaluation-harness, FP16 vs INT4.

Backends (one run = one model, results saved per run, compared afterwards):
  ours       llama/model.py + this repo's INT4 (R1 fp16, R2/R3 --group-size), or any
             fp16 state_dict via --state-dict (e.g. fake-quant weights from quant_improve.py)
  hf         HF transformers Llama-3.2-1B through lm-eval's own HFLM (R0, external reference)
  hf-torchao HF model + torchao Int4WeightOnlyConfig (R4, independent INT4 implementation).
             tinygemm int4 needs bf16, so compare R4 against `--backend hf --dtype bfloat16`.

The adapter only implements loglikelihood (multiple choice, LAMBADA, MMLU) and
loglikelihood_rolling; generate_until is not needed for the task list below.
Sequences are scored teacher-forced in one forward (M = sequence length > 16), so
the result depends on the quantized weights only, not on the dequant impl (v1..v4).

BOS: Llama 3 base expects <|begin_of_text|>. Both `ours` and `hf` prepend it
(hf: add_bos_token=True), so R0 and R1 see identical token ids.

Per-sample correctness is saved (log_samples) so two runs can be compared on the
same questions: paired bootstrap CI of the accuracy difference, and flips
(right -> wrong, wrong -> right; Dutta et al. 2024, "Accuracy is Not All You Need").

Usage (A100; datasets via HF_ENDPOINT=https://hf-mirror.com):
    python eval_tasks.py --name r1_fp16
    python eval_tasks.py --name r2_int4_g128 --group-size 128
    python eval_tasks.py --name r0_hf_fp16 --backend hf --hf-path /root/autodl-tmp/hf/Llama-3.2-1B
    python eval_tasks.py --name smoke --limit 20 --tasks arc_easy          # quick check
    python eval_tasks.py --compare r1_fp16 r2_int4_g128                   # CI + flips
"""
import argparse
import csv
import datetime
import json
import os
import time

import torch
import torch.nn.functional as F

OUT_DIR = "results/lm_eval"
ZERO_SHOT = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "lambada_openai"]
MAIN_METRIC = {"arc_easy": "acc_norm", "arc_challenge": "acc_norm", "hellaswag": "acc_norm"}  # else "acc"


# ---------------------------------------------------------------- scoring core
@torch.inference_mode()
def score_requests(model, reqs, token_budget=8192, max_len=2048):
    """reqs: list of (context_ids, continuation_ids). Returns [(sum logprob, is_greedy)].

    One forward per batch, right-padded (causal: padding after a sequence cannot
    change its logits). Batches are built longest-first under a tokens budget
    (B x L_max), because the model returns full fp32 logits [B, L, 128256].
    """
    items = []
    for i, (ctx, cont) in enumerate(reqs):
        assert len(cont) > 0 and len(ctx) > 0, "context must hold at least BOS"
        full = (ctx + cont)[-(max_len + 1):]
        inp = full[:-1]
        items.append((i, inp, len(cont)))
    items.sort(key=lambda t: -len(t[1]))

    out = [None] * len(reqs)
    b = 0
    while b < len(items):
        l_max = len(items[b][1])
        n = max(1, token_budget // l_max)
        batch = items[b:b + n]
        b += n
        x = torch.zeros(len(batch), l_max, dtype=torch.long, device="cuda")
        for r, (_, inp, _) in enumerate(batch):
            x[r, :len(inp)] = torch.tensor(inp, device="cuda")
        logits = model(x, 0)  # [B, L, V] fp32
        for r, (i, inp, n_cont) in enumerate(batch):
            end = len(inp)
            lp = F.log_softmax(logits[r, end - n_cont:end].float(), dim=-1)  # [n_cont, V]
            # targets: the continuation = last n_cont tokens of (inp + last token)
            full_tail = reqs[i][1][-n_cont:]
            tgt = torch.tensor(full_tail, device="cuda")
            out[i] = (lp.gather(1, tgt[:, None]).sum().item(), bool((lp.argmax(-1) == tgt).all()))
        del logits
    return out


# ---------------------------------------------------------------- lm-eval adapter
def make_lm(model, tokenizer, batch_tokens, max_len, add_bos=True):
    from lm_eval.api.model import TemplateLM
    from lm_eval import utils as lm_utils

    class OursLM(TemplateLM):
        def __init__(self):
            super().__init__()

        @property
        def eot_token_id(self):
            return tokenizer.eos_id

        @property
        def prefix_token_id(self):
            return tokenizer.bos_id

        def tok_encode(self, string, add_special_tokens=None, **kwargs):
            bos = add_bos if add_special_tokens is None else add_special_tokens
            return tokenizer.encode(string, bos=bos, eos=False)

        def tok_decode(self, tokens, **kwargs):
            return tokenizer.decode(tokens)

        def _loglikelihood_tokens(self, requests, disable_tqdm=False):
            return score_requests(model, [(c, k) for _, c, k in requests], batch_tokens, max_len)

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            res = []
            for (string,) in [r.args for r in requests]:
                ids = self.tok_encode(string, add_special_tokens=False)
                windows = [lm_utils.make_disjoint_window(w) for w in lm_utils.get_rolling_token_windows(
                    token_list=ids, prefix_token=self.prefix_token_id, max_seq_len=max_len, context_len=1)]
                res.append(sum(lp for lp, _ in score_requests(model, windows, batch_tokens, max_len)))
            return res

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError("generation tasks are not part of E1e")

    return OursLM()


def build_lm(a):
    if a.backend == "ours":
        import eval_ppl
        from llama.config import CKPT_DIR
        from llama.tokenizer import Tokenizer
        sd = torch.load(a.state_dict or os.path.join(CKPT_DIR, "consolidated.00.pth"),
                        map_location="cpu", weights_only=True, mmap=True)
        model = eval_ppl.build_model(sd, a.max_len, group_size=a.group_size)
        tok = Tokenizer(os.path.join(CKPT_DIR, "tokenizer.model"))
        return make_lm(model, tok, a.batch_tokens, a.max_len)

    from lm_eval.models.huggingface import HFLM
    dtype = getattr(torch, a.dtype)
    if a.backend == "hf":
        return HFLM(pretrained=a.hf_path, dtype=dtype, add_bos_token=True,
                    batch_size=a.hf_batch_size, max_length=a.max_len)
    # hf-torchao
    from transformers import AutoModelForCausalLM
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    m = AutoModelForCausalLM.from_pretrained(a.hf_path, torch_dtype=torch.bfloat16).cuda().eval()
    quantize_(m, Int4WeightOnlyConfig(group_size=a.group_size or 128))
    return HFLM(pretrained=m, tokenizer=a.hf_path, add_bos_token=True,
                batch_size=a.hf_batch_size, max_length=a.max_len)


# ---------------------------------------------------------------- run / compare
def per_sample(samples):
    """{task: {doc_id: {metric: 0/1}}} from lm-eval's log_samples output."""
    out = {}
    for task, rows in samples.items():
        out[task] = {}
        for r in rows:
            out[task][str(r["doc_id"])] = {k: float(r[k]) for k in ("acc", "acc_norm") if k in r}
    return out


def run(a):
    import lm_eval
    os.makedirs(OUT_DIR, exist_ok=True)
    lm = build_lm(a)
    tasks = a.tasks or ZERO_SHOT
    t0 = time.time()
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=None, limit=a.limit,
                                  log_samples=True, bootstrap_iters=1000)
    if a.mmlu:
        mm = lm_eval.simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=5, limit=a.limit,
                                     log_samples=True, bootstrap_iters=1000)
        res["results"].update(mm["results"])
        res["samples"].update(mm["samples"])
    dt = time.time() - t0

    rows, now = [], datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{a.name}] backend={a.backend} group={a.group_size} ({dt / 60:.1f} min)")
    for task, r in sorted(res["results"].items()):
        for key, val in r.items():
            if not isinstance(val, (int, float)) or "stderr" in key or "," not in key:
                continue
            metric = key.split(",")[0]
            if metric not in ("acc", "acc_norm", "perplexity"):
                continue
            se = r.get(f"{metric}_stderr,{key.split(',', 1)[1]}")
            print(f"  {task:<28} {metric:<10} {val:8.4f}" + (f" ± {se:.4f}" if isinstance(se, float) else ""))
            rows.append({"name": a.name, "backend": a.backend, "group_size": a.group_size or "",
                         "task": task, "metric": metric, "value": round(val, 6),
                         "stderr": round(se, 6) if isinstance(se, float) else "",
                         "limit": a.limit or "", "time": now})
    with open(os.path.join(OUT_DIR, f"{a.name}_samples.json"), "w") as f:
        json.dump(per_sample(res["samples"]), f)
    with open(os.path.join(OUT_DIR, f"{a.name}_results.json"), "w") as f:
        json.dump(res["results"], f, indent=1, default=str)
    path = os.path.join(OUT_DIR, "summary.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"saved {OUT_DIR}/{a.name}_samples.json, appended {len(rows)} rows to {path}")


def compare(ref_name, new_name, iters=10000, seed=0):
    """Paired comparison of two runs on the same questions."""
    load = lambda n: json.load(open(os.path.join(OUT_DIR, f"{n}_samples.json")))
    A, B = load(ref_name), load(new_name)
    g = torch.Generator().manual_seed(seed)
    print(f"{new_name} vs {ref_name}  (Δ = new − ref, percentage points; 95% paired bootstrap CI)")
    print(f"  {'task':<22} {'metric':<9} {'n':>6} {'ref':>7} {'new':>7} {'Δ':>7} {'95% CI':>17} "
          f"{'R→W':>6} {'W→R':>6}")
    groups = {}
    for task in sorted(set(A) & set(B)):
        key = "mmlu" if task.startswith("mmlu") else task
        groups.setdefault(key, []).append(task)
    for key, tasks in groups.items():
        a, b = [], []
        metric = MAIN_METRIC.get(key, "acc")
        for t in tasks:
            for d in set(A[t]) & set(B[t]):
                if metric in A[t][d] and metric in B[t][d]:
                    a.append(A[t][d][metric])
                    b.append(B[t][d][metric])
        if not a:
            continue
        a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
        n = len(a)
        idx = torch.randint(0, n, (iters, n), generator=g)
        diffs = (b[idx] - a[idx]).mean(1) * 100
        lo, hi = torch.quantile(diffs, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
        rw = ((a == 1) & (b == 0)).double().mean().item() * 100
        wr = ((a == 0) & (b == 1)).double().mean().item() * 100
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"  {key:<22} {metric:<9} {n:>6} {a.mean() * 100:7.2f} {b.mean() * 100:7.2f} "
              f"{(b - a).mean() * 100:+7.2f} [{lo:+6.2f}, {hi:+6.2f}]{sig} {rw:6.2f} {wr:6.2f}")
    print("  * = CI excludes 0.  R→W = right under ref, wrong under new (% of questions); W→R the reverse.")


def main():
    p = argparse.ArgumentParser(description="E1e: lm-eval downstream accuracy, FP16 vs INT4.")
    p.add_argument("--name", help="run name; outputs go to results/lm_eval/<name>_*")
    p.add_argument("--backend", default="ours", choices=["ours", "hf", "hf-torchao"])
    p.add_argument("--group-size", type=int, default=None, help="ours: INT4 RTN group size (omit = FP16)")
    p.add_argument("--state-dict", default=None, help="ours: load this fp16 state_dict instead of the checkpoint")
    p.add_argument("--hf-path", default="/root/autodl-tmp/hf/Llama-3.2-1B")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="hf backend dtype")
    p.add_argument("--tasks", nargs="+", default=None, help=f"default: {' '.join(ZERO_SHOT)}")
    p.add_argument("--mmlu", action="store_true", help="also run MMLU 5-shot")
    p.add_argument("--limit", type=int, default=None, help="max docs per task (smoke tests)")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--batch-tokens", type=int, default=8192, help="ours: tokens per forward (B x L)")
    p.add_argument("--hf-batch-size", default="auto")
    p.add_argument("--compare", nargs=2, metavar=("REF", "NEW"), help="compare two saved runs")
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        assert a.name, "--name is required"
        run(a)


if __name__ == "__main__":
    main()
