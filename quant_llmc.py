"""
E1f-B: GPTQ / AWQ with llm-compressor on the HF-format Llama-3.2-1B (runs in the
separate venv-quant, scripts/setup_vllm_env.sh). The output directory is a
compressed-tensors checkpoint that vLLM loads directly (Marlin INT4 kernels) and
that hf_ppl.py / eval_tasks.py --backend hf can evaluate.

Scheme W4A16_ASYM = 4-bit asymmetric, group 128 (same format family as our RTN).
lm_head is not quantized (llm-compressor's usual setting); compare with E1d's
`no_output` (RTN on the other 112 Linear).

Calibration: --nsamples random windows of --seqlen tokens from WikiText-2 *train*
(the test split is what we evaluate on; same domain, so also check Alpaca and lm-eval).

Usage (venv-quant):
    HF_ENDPOINT=https://hf-mirror.com /root/autodl-tmp/venv-quant/bin/python quant_llmc.py --method gptq
    HF_ENDPOINT=https://hf-mirror.com /root/autodl-tmp/venv-quant/bin/python quant_llmc.py --method awq
"""
import argparse
import os
import random
import time

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def calib_dataset(tok, nsamples, seqlen, seed):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    rng = random.Random(seed)
    starts = [rng.randrange(0, len(ids) - seqlen) for _ in range(nsamples)]
    rows = [ids[s:s + seqlen].tolist() for s in starts]
    print(f"calibration: {nsamples} x {seqlen} tokens from WikiText-2 train ({len(ids)} tokens)", flush=True)
    return Dataset.from_dict({"input_ids": rows, "attention_mask": [[1] * seqlen for _ in rows]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["gptq", "awq"])
    p.add_argument("--scheme", default="W4A16_ASYM")
    p.add_argument("--hf-path", default="/root/autodl-tmp/hf/Llama-3.2-1B")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", default="/root/autodl-tmp/llmc")
    a = p.parse_args()
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import GPTQModifier
    import llmcompressor

    out = os.path.join(a.out_root, f"{a.method}_{a.scheme.lower()}")
    print(f"llmcompressor {llmcompressor.__version__} | torch {torch.__version__} | {a.method} {a.scheme} -> {out}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.hf_path)
    model = AutoModelForCausalLM.from_pretrained(a.hf_path, dtype=torch.float16, device_map="cuda")
    ds = calib_dataset(tok, a.nsamples, a.seqlen, a.seed)
    if a.method == "gptq":
        recipe = GPTQModifier(targets="Linear", scheme=a.scheme, ignore=["lm_head"])
    else:
        recipe = AWQModifier(targets=["Linear"], scheme=a.scheme, ignore=["lm_head"])
    t0 = time.time()
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=a.seqlen, num_calibration_samples=a.nsamples)
    print(f"oneshot done in {time.time() - t0:.0f} s", flush=True)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"saved {out} ({size / 2**20:.0f} MB on disk)", flush=True)


if __name__ == "__main__":
    main()
