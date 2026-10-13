"""
Benchmark FP16 vs INT4 quantization and dense vs MoE (slice / LoRA).

Run with: CUDA_VISIBLE_DEVICES=0 python run_benchmark.py
"""
import os
import time
import torch

from llama.config import CKPT_DIR
from llama.model import ModelArgs, Llama
from llama.tokenizer import Tokenizer


def load_model(max_batch_size=4, max_seq_len=320, device="cuda"):
    checkpoint_dir = CKPT_DIR
    tokenizer = Tokenizer(os.path.join(checkpoint_dir, "tokenizer.model"))
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "consolidated.00.pth"),
        map_location="cpu", weights_only=True,
    )
    model_args = ModelArgs()
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_prompt_of_length(base_prompt, desired_len, tokenizer):
    tokens = tokenizer.encode(base_prompt, bos=True, eos=False)
    while len(tokens) < desired_len:
        tokens += tokenizer.encode(" " + base_prompt, bos=False, eos=False)
    return tokenizer.decode(tokens[:desired_len])


def benchmark_generation(model, tokenizer, batch_size, input_len, output_len, label, device="cuda"):
    base = "Once upon a time in a galaxy far away"
    prompts = [generate_prompt_of_length(base, input_len, tokenizer) for _ in range(batch_size)]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()

    results = model.generate(
        tokenizer, prompts, max_gen_len=output_len,
        temperature=0.6, top_p=0.9, kv_caching=True, device=device,
    )

    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"  [{label}] batch={batch_size} in={input_len} out={output_len}: "
          f"{elapsed:.2f}s, {peak_mem:.0f} MB peak, "
          f"{batch_size * output_len / elapsed:.1f} tok/s")
    return elapsed, peak_mem


def benchmark_quant():
    """Benchmark FP16 vs INT4 weight-only quantization."""
    print("\n" + "=" * 70)
    print("INT4 Quantization Benchmark")
    print("=" * 70)

    from llama.quantize import quantize_model, print_model_size

    configs = [(1, 256, 32), (8, 256, 32), (16, 256, 32)]

    # FP16 benchmarks
    print("\n--- FP16 Baseline ---")
    for bs, il, ol in configs:
        torch.cuda.empty_cache()
        model, tokenizer = load_model(max_batch_size=bs, max_seq_len=il + ol)
        benchmark_generation(model, tokenizer, bs, il, ol, "FP16")
        del model
        torch.cuda.empty_cache()

    # INT4 benchmarks
    print("\n--- INT4 Quantized ---")
    for bs, il, ol in configs:
        torch.cuda.empty_cache()
        model, tokenizer = load_model(max_batch_size=bs, max_seq_len=il + ol)
        print_model_size(model)
        quantize_model(model, group_size=128)
        print_model_size(model)
        benchmark_generation(model, tokenizer, bs, il, ol, "INT4")
        del model
        torch.cuda.empty_cache()


def benchmark_moe():
    """Benchmark dense vs MoE."""
    print("\n" + "=" * 70)
    print("MoE Benchmark")
    print("=" * 70)

    from llama.moe import convert_to_moe, get_expert_load_stats

    # Dense baseline
    print("\n--- Dense Baseline ---")
    torch.cuda.empty_cache()
    model, tokenizer = load_model(max_batch_size=1, max_seq_len=320)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    benchmark_generation(model, tokenizer, 1, 256, 32, "Dense")
    del model
    torch.cuda.empty_cache()

    for init_mode in ["slice", "lora"]:
        print(f"\n--- MoE (N=4, K=2, init={init_mode}) ---")
        torch.cuda.empty_cache()
        model, tokenizer = load_model(max_batch_size=1, max_seq_len=320)
        model, trainable_params = convert_to_moe(
            model, num_experts=4, top_k=2, init_mode=init_mode
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_count:,}")

        benchmark_generation(model, tokenizer, 1, 256, 32, f"MoE-{init_mode}")

        prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming every industry.",
            "What is the meaning of life?",
            "The weather today is sunny and warm.",
        ]
        stats = get_expert_load_stats(model, tokenizer, prompts)
        print("\n  Expert load distribution:")
        for layer_idx in sorted(stats.keys())[:4]:
            counts = stats[layer_idx]
            total = counts.sum().item()
            pcts = [f"{c.item()/total*100:.0f}%" for c in counts]
            print(f"    Layer {layer_idx}: {pcts} (total={total})")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    benchmark_quant()
    benchmark_moe()

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
