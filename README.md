# llama-efficient-inference

Efficient inference experiments on Llama 3.2-1B in PyTorch:

- **INT4 weight-only quantization** — per-group (128) asymmetric INT4, two values packed per `uint8`, dequantized to FP16 before each matmul (`llama/quantize.py`).
- **Mixture-of-Experts FFN** — top-K routed experts built from the dense FFN, two variants (`llama/moe.py`):
  - `slice`: split the pretrained FFN into N smaller experts.
  - `lora`: keep the dense FFN frozen and route among N LoRA adapters (B zero-initialized, so the converted model starts identical to dense).

Model code is based on Meta's [Llama 3 reference implementation](https://github.com/meta-llama/llama3).

## Setup

```bash
pip install -r requirements.txt

# Weights (requires access to meta-llama/Llama-3.2-1B on Hugging Face)
huggingface-cli download meta-llama/Llama-3.2-1B --include "original/*" --local-dir ./hf
mkdir -p ckpt && mv hf/original/* ckpt/

# Fine-tuning / eval data (needs `pip install datasets`)
python prepare_data.py --output-dir ./data
```

Paths default to `./ckpt` and `./data`; override with `LLAMA_CKPT_DIR` and `DATA_DIR` (see `llama/config.py`).

## Run

| Command | What it does |
|---|---|
| `python inference.py` | Smoke test: load the model and generate |
| `python check_impl.py` | Correctness checks for quantization and MoE |
| `python bench.py --mode {fp16,int4,moe-slice,moe-lora} --bs 1 8 16` | Prefill / per-step decode latency and peak memory (warmup, CUDA events, CSV to `results/`) |
| `python mem_snapshot.py --mode int4 --bs 1` | What is alive on the GPU at peak memory (PyTorch memory snapshot, viewable at pytorch.org/memory_viz) |
| `python run_benchmark.py` | FP16 vs INT4, dense vs MoE: end-to-end `generate()` runtime and peak memory |
| `python train_moe.py --init-mode {slice,lora}` | Fine-tune the MoE on Alpaca; writes `checkpoints/moe_finetuned.pt` |
| `python eval_moe.py --init-mode {slice,lora}` | Perplexity, next-token accuracy, throughput, expert load |
