"""
T8: vLLM as the "mature system" reference (runs in venv-vllm). Same workload as
bench.py: input_len random token ids, output_len tokens, greedy, ignore_eos, one
batch of `bs` identical-length requests.

  prefill_ms  median wall time of generate(max_tokens=1)
  e2e_ms      median wall time of generate(max_tokens=output_len)
  decode_ms   (e2e_ms - prefill_ms) / (output_len - 1)   per step for the whole batch

vLLM defaults stay on (CUDA graphs, its own kernels; INT4 checkpoints from
quant_llmc.py go through vLLM's compressed-tensors W4A16 path). The kernel vLLM
picks is printed in its startup log. Run with the GPU otherwise idle.

Usage (venv-vllm):
    /root/autodl-tmp/venv-vllm/bin/python vllm_bench.py --model fp16=/root/autodl-tmp/hf/Llama-3.2-1B
"""
import argparse
import csv
import datetime
import os
import statistics
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="name=path")
    p.add_argument("--bs", type=int, nargs="+", default=[1, 8, 16])
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--gpu-mem", type=float, default=0.5)
    p.add_argument("--tokenizer", default="/root/autodl-tmp/hf/Llama-3.2-1B",
                   help="checkpoints saved by venv-quant (transformers 5.x) write a tokenizer class vLLM's transformers does not know")
    p.add_argument("--csv", default="results/vllm_bench.csv")
    a = p.parse_args()
    from vllm import LLM, SamplingParams
    import vllm
    name, path = a.model.split("=", 1)
    llm = LLM(model=path, tokenizer=a.tokenizer, dtype="float16", max_model_len=a.input_len + a.output_len + 16,
              gpu_memory_utilization=a.gpu_mem, seed=0)
    g = torch.Generator().manual_seed(0)
    rows = []
    for bs in a.bs:
        prompts = [{"prompt_token_ids": torch.randint(0, 128000, (a.input_len,), generator=g).tolist()} for _ in range(bs)]
        def timed(max_tokens):
            sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
            for _ in range(a.warmup):
                llm.generate(prompts, sp, use_tqdm=False)
            ts = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                out = llm.generate(prompts, sp, use_tqdm=False)
                ts.append((time.perf_counter() - t0) * 1e3)
            assert all(len(o.outputs[0].token_ids) == max_tokens for o in out)
            return statistics.median(ts)
        pre, e2e = timed(1), timed(a.output_len)
        dec = (e2e - pre) / (a.output_len - 1)
        print(f"[vllm {name}] bs={bs}: prefill {pre:.2f} ms | e2e {e2e:.1f} ms | decode {dec:.3f} ms/step", flush=True)
        rows.append({"model": name, "path": path, "bs": bs, "input_len": a.input_len, "output_len": a.output_len,
                     "prefill_ms": round(pre, 3), "e2e_ms": round(e2e, 3), "decode_ms": round(dec, 3),
                     "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                     "time": datetime.datetime.now().isoformat(timespec="seconds")})
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
