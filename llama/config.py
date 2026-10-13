"""Paths to the Llama 3.2-1B checkpoint and the fine-tuning data.

Override with environment variables:
    export LLAMA_CKPT_DIR=/path/to/Llama3.2-1B   # consolidated.00.pth, tokenizer.model
    export DATA_DIR=/path/to/data                # alpaca_500.json, mt_bench_prompts.json
"""
import os

CKPT_DIR = os.environ.get("LLAMA_CKPT_DIR", "./ckpt")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
