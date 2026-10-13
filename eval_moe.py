"""
Quantitative evaluation of MoE vs Dense:
1. Perplexity on held-out data (Alpaca samples 200-299, disjoint from the
   first 200 samples used by train_moe.py)
2. Next-token prediction accuracy
3. Inference speed comparison

Usage: CUDA_VISIBLE_DEVICES=0 python eval_moe.py
"""
import argparse
import os, json, time, math, gc, torch
import torch.nn as nn

from llama.config import CKPT_DIR, DATA_DIR
from llama.model import ModelArgs, Llama
from llama.tokenizer import Tokenizer
from llama.moe import convert_to_moe, get_expert_load_stats


def load_model(device="cuda", max_seq_len=256, kv_caching=True):
    checkpoint_dir = CKPT_DIR
    tokenizer = Tokenizer(os.path.join(checkpoint_dir, "tokenizer.model"))
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "consolidated.00.pth"),
        map_location="cpu", weights_only=True,
    )
    model_args = ModelArgs()
    model_args.max_batch_size = 1
    model_args.max_seq_len = max_seq_len
    model_args.kv_caching = kv_caching
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, texts, device="cuda", max_len=128):
    """Compute perplexity and top-1 accuracy on a list of texts."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for text in texts:
        tokens = tokenizer.encode(text, bos=True, eos=True)
        if len(tokens) < 4:
            continue
        tokens = tokens[:max_len]

        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

        logits = model(input_ids, start_pos=0)  # (1, S, vocab)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        preds = logits.argmax(dim=-1).squeeze(0)  # (S,)
        correct = (preds == target_ids.squeeze(0)).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_tokens += target_ids.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    acc = total_correct / total_tokens * 100
    return ppl, acc, total_tokens


@torch.no_grad()
def evaluate_generation_speed(model, tokenizer, prompts, device="cuda", max_gen_len=32):
    """Measure generation speed (tokens/sec)."""
    model.eval()
    torch.cuda.synchronize()
    t0 = time.time()

    total_tokens = 0
    for prompt in prompts:
        results = model.generate(
            tokenizer, [prompt], max_gen_len=max_gen_len,
            temperature=0.6, top_p=0.9, kv_caching=True, device=device,
        )
        total_tokens += len(tokenizer.encode(results[0]["generation"], bos=False, eos=False))

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tok_per_sec = total_tokens / elapsed
    return elapsed, tok_per_sec, total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dense vs MoE models.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--init-mode", choices=["slice", "lora"], default="slice")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    torch.manual_seed(42)

    # Held-out split: train_moe.py uses the first 200 Alpaca
    # samples for training; we evaluate on samples 200-299 here.
    with open(os.path.join(DATA_DIR, "alpaca_500.json")) as f:
        alpaca_data = json.load(f)
    eval_texts = [s["text"] for s in alpaca_data[200:300]]  # held-out from training (first 200 used for training)

    with open(os.path.join(DATA_DIR, "mt_bench_prompts.json")) as f:
        mt_prompts = json.load(f)

    print("=" * 70)
    print("Quantitative MoE Evaluation")
    print(f"Eval set: {len(eval_texts)} held-out Alpaca samples")
    print("=" * 70)

    # ================================================================
    # 1. Dense Baseline
    # ================================================================
    print("\n[1/3] Dense Baseline...")
    model, tokenizer = load_model(device=device, max_seq_len=160, kv_caching=False)
    ppl_dense, acc_dense, n_tok = evaluate_perplexity(model, tokenizer, eval_texts, device)
    print(f"  Perplexity: {ppl_dense:.2f}")
    print(f"  Next-token accuracy: {acc_dense:.2f}%")
    print(f"  Evaluated on {n_tok} tokens")

    # Speed test (needs KV caching)
    del model; gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = load_model(device=device, max_seq_len=256, kv_caching=True)
    elapsed, tps, _ = evaluate_generation_speed(model, tokenizer, mt_prompts[:10], device)
    print(f"  Generation speed: {tps:.1f} tok/s ({elapsed:.2f}s for 10 prompts)")
    speed_dense = tps

    del model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # 2. MoE Before Training
    # ================================================================
    print(f"\n[2/3] MoE Before Training (init={args.init_mode}, N={args.num_experts}, K={args.top_k})...")
    model, tokenizer = load_model(device=device, max_seq_len=160, kv_caching=False)
    model, _ = convert_to_moe(
        model, num_experts=args.num_experts, top_k=args.top_k,
        init_mode=args.init_mode, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    ppl_moe_before, acc_moe_before, _ = evaluate_perplexity(model, tokenizer, eval_texts, device)
    print(f"  Perplexity: {ppl_moe_before:.2f}")
    print(f"  Next-token accuracy: {acc_moe_before:.2f}%")

    del model; gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = load_model(device=device, max_seq_len=256, kv_caching=True)
    model, _ = convert_to_moe(
        model, num_experts=args.num_experts, top_k=args.top_k,
        init_mode=args.init_mode, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    elapsed, tps, _ = evaluate_generation_speed(model, tokenizer, mt_prompts[:10], device)
    print(f"  Generation speed: {tps:.1f} tok/s ({elapsed:.2f}s for 10 prompts)")
    speed_moe_before = tps

    del model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # 3. MoE After Training
    # ================================================================
    print("\n[3/3] MoE After Training...")
    # Load and convert, then load fine-tuned weights
    model, tokenizer = load_model(device=device, max_seq_len=160, kv_caching=False)
    model, _ = convert_to_moe(
        model, num_experts=args.num_experts, top_k=args.top_k,
        init_mode=args.init_mode, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    ckpt_path = "checkpoints/moe_finetuned.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if (
            ckpt.get("num_experts", args.num_experts) != args.num_experts
            or ckpt.get("top_k", args.top_k) != args.top_k
            or ckpt.get("init_mode", args.init_mode) != args.init_mode
        ):
            print(
                "  WARNING: checkpoint config does not match current eval config "
                f"(ckpt: N={ckpt.get('num_experts')}, K={ckpt.get('top_k')}, init={ckpt.get('init_mode')})"
            )
        model.load_state_dict(ckpt["moe_state"], strict=False)
        model.to(device)
        print(f"  Loaded fine-tuned weights from {ckpt_path}")
        print(f"  Training losses: {ckpt.get('losses', 'N/A')}")
    else:
        print("  WARNING: No fine-tuned checkpoint found, using un-trained MoE")

    ppl_moe_after, acc_moe_after, _ = evaluate_perplexity(model, tokenizer, eval_texts, device)
    print(f"  Perplexity: {ppl_moe_after:.2f}")
    print(f"  Next-token accuracy: {acc_moe_after:.2f}%")

    del model; gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = load_model(device=device, max_seq_len=256, kv_caching=True)
    model, _ = convert_to_moe(
        model, num_experts=args.num_experts, top_k=args.top_k,
        init_mode=args.init_mode, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["moe_state"], strict=False)
        model.to(device)
    elapsed, tps, _ = evaluate_generation_speed(model, tokenizer, mt_prompts[:10], device)
    print(f"  Generation speed: {tps:.1f} tok/s ({elapsed:.2f}s for 10 prompts)")
    speed_moe_after = tps

    # Expert load stats
    stats = get_expert_load_stats(model, tokenizer, mt_prompts[:10], device=device)
    print("  Expert load (first 4 layers):")
    for li in sorted(stats.keys())[:4]:
        c = stats[li]
        total = c.sum().item()
        pcts = [f"{x.item()/total*100:.0f}%" for x in c]
        print(f"    Layer {li}: {pcts}")

    del model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Summary Table
    # ================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} | {'Dense':>12} | {'MoE (before)':>12} | {'MoE (after)':>12}")
    print("-" * 70)
    print(f"{'Perplexity ↓':<25} | {ppl_dense:>12.2f} | {ppl_moe_before:>12.2f} | {ppl_moe_after:>12.2f}")
    print(f"{'Next-token Acc ↑':<25} | {acc_dense:>11.2f}% | {acc_moe_before:>11.2f}% | {acc_moe_after:>11.2f}%")
    print(f"{'Gen Speed (tok/s) ↑':<25} | {speed_dense:>12.1f} | {speed_moe_before:>12.1f} | {speed_moe_after:>12.1f}")

    # Improvement summary
    print(f"\nMoE-before vs Dense:")
    print(f"  PPL change: {ppl_dense:.2f} → {ppl_moe_before:.2f} ({'↑ worse' if ppl_moe_before > ppl_dense else '↓ better'})")
    print(f"  Acc change: {acc_dense:.2f}% → {acc_moe_before:.2f}%")
    print(f"\nMoE-after vs MoE-before:")
    print(f"  PPL change: {ppl_moe_before:.2f} → {ppl_moe_after:.2f} ({'↑ worse' if ppl_moe_after > ppl_moe_before else '↓ better'})")
    print(f"  Acc change: {acc_moe_before:.2f}% → {acc_moe_after:.2f}%")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
