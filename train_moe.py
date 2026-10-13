"""
Fine-tune the MoE router (and optionally expert / LoRA weights) on an
Alpaca subset, then qualitatively evaluate on MT-bench prompts.

Usage: CUDA_VISIBLE_DEVICES=0 python train_moe.py
"""

import argparse
import os, json, time, gc, torch
import torch.nn as nn

from llama.config import CKPT_DIR, DATA_DIR
from llama.model import ModelArgs, Llama
from llama.tokenizer import Tokenizer
from llama.moe import (
    convert_to_moe,
    get_expert_load_stats,
    MoEFeedForward,
    LoRAMoEFeedForward,
)


def load_model(device="cuda", max_seq_len=256, kv_caching=True):
    checkpoint_dir = CKPT_DIR
    tokenizer = Tokenizer(os.path.join(checkpoint_dir, "tokenizer.model"))
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "consolidated.00.pth"),
        map_location="cpu",
        weights_only=True,
    )
    model_args = ModelArgs()
    model_args.max_batch_size = 1
    model_args.max_seq_len = max_seq_len
    model_args.kv_caching = kv_caching
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    gc.collect()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.to(device)
    return model, tokenizer


def evaluate_on_prompts(
    model, tokenizer, prompts, label, device="cuda", max_gen_len=64
):
    """Generate outputs for a list of prompts and show samples."""
    print(f"\n--- {label} ---")
    model.eval()
    outputs = []
    for prompt in prompts:
        results = model.generate(
            tokenizer,
            [prompt],
            max_gen_len=max_gen_len,
            temperature=0.6,
            top_p=0.9,
            kv_caching=True,
            device=device,
        )
        outputs.append(results[0]["generation"])

    # Show 5 samples
    for i in range(min(5, len(prompts))):
        print(f"  Q: {prompts[i][:80]}")
        print(f"  A: {outputs[i][:120]}")
        print()
    return outputs


def train_moe_router(
    model,
    tokenizer,
    train_texts,
    device="cuda",
    lr_router=1e-4,
    lr_expert=1e-5,
    epochs=3,
    max_seq_len=128,
    unfreeze_experts=True,
):
    """Fine-tune MoE router (and optionally experts/LoRA adapters) on text data."""

    # Freeze everything except router and (optionally) expert weights
    for param in model.parameters():
        param.requires_grad = False

    param_groups = []
    for layer in model.layers:
        ff = layer.feed_forward

        if isinstance(ff, LoRAMoEFeedForward):
            # LoRA mode: router + LoRA adapters are always trainable
            for p in ff.router.parameters():
                p.requires_grad = True
            param_groups.append(
                {"params": list(ff.router.parameters()), "lr": lr_router}
            )

            lora_params = []
            for expert in ff.lora_experts:
                for p in expert.parameters():
                    p.requires_grad = True
                    lora_params.append(p)
            param_groups.append({"params": lora_params, "lr": lr_expert})

        elif isinstance(ff, MoEFeedForward):
            # Slice mode: router + optionally full expert weights
            for p in ff.router.parameters():
                p.requires_grad = True
            param_groups.append(
                {"params": list(ff.router.parameters()), "lr": lr_router}
            )

            if unfreeze_experts:
                expert_params = []
                for ew in [ff.expert_w1, ff.expert_w2, ff.expert_w3]:
                    for expert in ew:
                        for p in expert.parameters():
                            p.requires_grad = True
                            expert_params.append(p)
                param_groups.append({"params": expert_params, "lr": lr_expert})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        n_tokens = 0
        t0 = time.time()

        for i, text in enumerate(train_texts):
            tokens = tokenizer.encode(text, bos=True, eos=True)
            if len(tokens) < 4:
                continue
            tokens = tokens[:max_seq_len]

            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            # Forward pass — model.forward already returns float32 logits
            logits = model(input_ids, start_pos=0)  # (1, S, vocab)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            loss_val = loss.item()
            if not (loss_val != loss_val):  # check for NaN
                epoch_loss += loss_val * target_ids.numel()
                n_tokens += target_ids.numel()

            if (i + 1) % 50 == 0:
                avg = epoch_loss / max(n_tokens, 1)
                print(
                    f"  Epoch {epoch+1}, step {i+1}/{len(train_texts)}, loss={avg:.4f}"
                )

        avg_loss = epoch_loss / max(n_tokens, 1)
        elapsed = time.time() - t0
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, time={elapsed:.1f}s")

    return losses


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MoE on Alpaca subset.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--init-mode", choices=["slice", "lora"], default="slice")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--train-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr-router", type=float, default=1e-3)
    parser.add_argument("--lr-expert", type=float, default=1e-4)
    parser.add_argument("--unfreeze-experts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", default="checkpoints/moe_finetuned.pt",
                        help="where to save the MoE weights (eval_moe.py --ckpt reads it)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    # Data split:
    # the first 200 Alpaca samples are used for training; eval_moe.py uses
    # samples 200-299 as a held-out set.
    with open(os.path.join(DATA_DIR, "alpaca_500.json")) as f:
        alpaca_data = json.load(f)
    train_texts = [s["text"] for s in alpaca_data]
    with open(os.path.join(DATA_DIR, "mt_bench_prompts.json")) as f:
        mt_prompts = json.load(f)

    # ================================================================
    # 1. Dense baseline evaluation
    # ================================================================
    print("=" * 70)
    print("MoE Fine-tuning Experiment")
    print("=" * 70)
    print(
        f"Config: N={args.num_experts}, K={args.top_k}, init_mode={args.init_mode}, "
        f"train_samples={args.train_samples}, epochs={args.epochs}, seed={args.seed}"
    )

    print("\n[1/4] Loading dense baseline model...")
    model, tokenizer = load_model(device=device, max_seq_len=256)
    dense_outputs = evaluate_on_prompts(model, tokenizer, mt_prompts, "Dense Baseline")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ================================================================
    # 2. MoE before fine-tuning
    # ================================================================
    print("\n[2/4] Converting to MoE (before fine-tuning)...")
    model, tokenizer = load_model(device=device, max_seq_len=256)
    model, router_params = convert_to_moe(
        model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        init_mode=args.init_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    moe_before_outputs = evaluate_on_prompts(
        model, tokenizer, mt_prompts, "MoE (Before Training)"
    )

    # Expert load stats before training
    stats_before = get_expert_load_stats(
        model, tokenizer, mt_prompts[:10], device=device
    )
    print("Expert load before training (first 4 layers):")
    for li in sorted(stats_before.keys())[:4]:
        c = stats_before[li]
        total = c.sum().item()
        pcts = [f"{x.item()/total*100:.0f}%" for x in c]
        print(f"  Layer {li}: {pcts}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ================================================================
    # 3. Fine-tune MoE
    # ================================================================
    print("\n[3/4] Fine-tuning MoE router + experts...")
    # Training uses kv_caching=False because we run full-sequence next-token
    # prediction, not incremental autoregressive decoding.
    model, tokenizer = load_model(device=device, max_seq_len=160, kv_caching=False)
    model, router_params = convert_to_moe(
        model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        init_mode=args.init_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    losses = train_moe_router(
        model,
        tokenizer,
        train_texts[: args.train_samples],
        device=device,
        lr_router=args.lr_router,
        lr_expert=args.lr_expert,
        epochs=args.epochs,
        max_seq_len=64,
        unfreeze_experts=args.unfreeze_experts,
    )

    # Save only the MoE-related parameters. The dense base model is reloaded
    # from the original checkpoint during evaluation.
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    moe_state = {}
    for k, v in model.state_dict().items():
        if "feed_forward" in k:
            moe_state[k] = v.cpu()
    torch.save(
        {
            "losses": losses,
            "moe_state": moe_state,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "init_mode": args.init_mode,
        },
        args.ckpt,
    )
    print(f"Training losses: {losses}")

    # ================================================================
    # 4. Evaluate fine-tuned MoE
    # ================================================================
    print("\n[4/4] Evaluating fine-tuned MoE...")
    # Reload with larger seq_len for generation
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = load_model(device=device, max_seq_len=256)
    model, _ = convert_to_moe(
        model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        init_mode=args.init_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # strict=False is intentional: this checkpoint only contains the MoE /
    # feed_forward subset, not a full dense-model state_dict.
    ckpt = torch.load(
        args.ckpt, map_location="cpu", weights_only=True
    )
    model.load_state_dict(ckpt["moe_state"], strict=False)
    model.to(device)

    moe_after_outputs = evaluate_on_prompts(
        model, tokenizer, mt_prompts, "MoE (After Training)"
    )

    # Expert load stats after training
    stats_after = get_expert_load_stats(
        model, tokenizer, mt_prompts[:10], device=device
    )
    print("Expert load after training (first 4 layers):")
    for li in sorted(stats_after.keys())[:4]:
        c = stats_after[li]
        total = c.sum().item()
        pcts = [f"{x.item()/total*100:.0f}%" for x in c]
        print(f"  Layer {li}: {pcts}")

    # ================================================================
    # Summary comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Side-by-side on 5 prompts")
    print("=" * 70)
    for i in range(min(5, len(mt_prompts))):
        print(f"\nQ: {mt_prompts[i][:80]}")
        print(f"  Dense:      {dense_outputs[i][:100]}")
        print(f"  MoE-before: {moe_before_outputs[i][:100]}")
        print(f"  MoE-after:  {moe_after_outputs[i][:100]}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
