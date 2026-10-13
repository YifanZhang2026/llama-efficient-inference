"""
check_impl.py — correctness checks for llama/quantize.py and llama/moe.py.

Run:  CUDA_VISIBLE_DEVICES=0 python check_impl.py

This script runs structural and numerical correctness checks on
llama/quantize.py and llama/moe.py.  Each check prints [PASS], [FAIL], or
[SKIP] with a one-line diagnosis.  It verifies correctness, NOT performance
(see run_benchmark.py for that).  Unimplemented parts show up as SKIP.
"""
import os
import traceback
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama.config import CKPT_DIR
from llama.model import ModelArgs, Llama
from llama.tokenizer import Tokenizer

CHECKPOINT_DIR = CKPT_DIR

DIM = 2048         # Llama-3.2-1B
HIDDEN_DIM = 8192  # Llama-3.2-1B FFN hidden size
N_LAYERS = 16

_results = {"pass": 0, "fail": 0, "skip": 0}


@contextmanager
def check(name):
    print(f"  [ RUN ] {name}")
    try:
        yield
        print(f"  [PASS] {name}")
        _results["pass"] += 1
    except NotImplementedError as e:
        msg = str(e) or "not implemented yet"
        print(f"  [SKIP] {name}: {msg}")
        _results["skip"] += 1
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        _results["fail"] += 1
    except Exception as e:
        print(f"  [FAIL] {name}: unexpected {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        _results["fail"] += 1


def load_fresh_model(max_seq_len=128):
    """Load a fresh Llama-3.2-1B model in FP16 on CUDA."""
    tokenizer = Tokenizer(os.path.join(CHECKPOINT_DIR, "tokenizer.model"))
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "consolidated.00.pth"),
        map_location="cpu", weights_only=True,
    )
    args = ModelArgs()
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Llama(args)
    model.load_state_dict(ckpt, strict=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.to("cuda").eval()
    return model, tokenizer


# ===================================================================
# INT4 Quantization
# ===================================================================

def quant_checks():
    print("\n" + "=" * 70)
    print("INT4 Quantization (llama/quantize.py)")
    print("=" * 70)

    try:
        from llama.quantize import QuantizedLinear, quantize_model
    except ImportError as e:
        print(f"  [SKIP] cannot import llama.quantize: {e}")
        _results["skip"] += 4
        return

    with check("quantize_tensor returns correct shapes / dtypes"):
        out_f, in_f, gs = 128, 256, 32
        w = torch.randn(out_f, in_f, dtype=torch.float16, device="cuda")
        packed, scale, zp = QuantizedLinear.quantize_tensor(w, gs)
        assert packed.shape == (out_f, in_f // 2), f"packed shape {packed.shape} != ({out_f}, {in_f // 2})"
        assert packed.dtype == torch.uint8, f"packed dtype {packed.dtype} != uint8"
        assert scale.shape == (out_f, in_f // gs), f"scale shape {scale.shape}"
        assert zp.shape == (out_f, in_f // gs), f"zero_point shape {zp.shape}"

    with check("quantize → dequantize round-trip has bounded error"):
        out_f, in_f, gs = 256, 512, 128
        w = torch.randn(out_f, in_f, dtype=torch.float16, device="cuda") * 0.1
        packed, scale, zp = QuantizedLinear.quantize_tensor(w, gs)
        w_deq = QuantizedLinear.dequantize_packed(packed, scale, zp, gs)
        max_err = (w.float() - w_deq.float()).abs().max().item()
        rel = max_err / (w.abs().max().item() + 1e-8)
        assert rel > 0, "round-trip error is 0 — you may not actually be quantizing"
        assert rel < 0.2, f"round-trip rel-error {rel:.3f} too large — packing or scale bug"

    with check("quantize_model() reduces model size by ≈60%"):
        model, _ = load_fresh_model()
        size_before = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for b in model.buffers())
        ) / 1024**2
        quantize_model(model, group_size=128)
        size_after = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for b in model.buffers())
        ) / 1024**2
        reduction = (1 - size_after / size_before) * 100
        assert 50 <= reduction <= 70, (
            f"size reduction {reduction:.1f}% outside [50%, 70%] "
            f"(before={size_before:.0f}MB, after={size_after:.0f}MB)"
        )
        del model
        torch.cuda.empty_cache()

    with check("quantized model generates coherent (non-gibberish) text"):
        model, tokenizer = load_fresh_model()
        quantize_model(model, group_size=128)
        results = model.generate(
            tokenizer, ["The meaning of life is"], max_gen_len=16,
            temperature=0.6, top_p=0.9, kv_caching=True, device="cuda",
        )
        out = results[0]["generation"]
        assert len(out) > 0, "empty generation"
        alpha_ratio = sum(c.isalpha() for c in out) / max(len(out), 1)
        assert alpha_ratio > 0.3, f"output is {alpha_ratio:.0%} alphabetic — looks like gibberish: {out!r}"
        del model
        torch.cuda.empty_cache()


# ===================================================================
# MoE
# ===================================================================

def moe_checks():
    print("\n" + "=" * 70)
    print("Mixture-of-Experts (llama/moe.py)")
    print("=" * 70)

    try:
        from llama.moe import (
            MoEFeedForward, LoRAExpert, LoRAMoEFeedForward,
            convert_to_moe, get_expert_load_stats,
        )
    except ImportError as e:
        print(f"  [SKIP] cannot import llama.moe: {e}")
        _results["skip"] += 9
        return

    # ---- slice-mode unit checks ----

    with check("MoEFeedForward.forward preserves input shape"):
        moe = MoEFeedForward(dim=DIM, hidden_dim=HIDDEN_DIM, num_experts=4, top_k=2).to("cuda").half()
        x = torch.randn(1, 8, DIM, device="cuda", dtype=torch.float16)
        y = moe(x)
        assert y.shape == x.shape, f"output shape {y.shape} != input {x.shape}"

    with check("top-K routing weights sum to 1 after renormalization"):
        moe = MoEFeedForward(dim=DIM, hidden_dim=HIDDEN_DIM, num_experts=4, top_k=2).to("cuda").half()
        x = torch.randn(1, 16, DIM, device="cuda", dtype=torch.float16)
        _ = moe(x)
        # recompute the top-k weights the forward *should* have produced
        x_flat = x.reshape(-1, DIM)
        logits = moe.router(x_flat)
        weights = F.softmax(logits.float(), dim=-1)
        tk_w, _ = torch.topk(weights, 2, dim=-1)
        tk_w = tk_w / tk_w.sum(dim=-1, keepdim=True)
        assert torch.allclose(tk_w.sum(dim=-1), torch.ones(16, device="cuda"), atol=1e-3), \
            "top-k weights don't sum to 1 per token"

    # ---- LoRA unit checks ----

    with check("LoRAExpert outputs exactly zero at initialization"):
        expert = LoRAExpert(dim=DIM, rank=8, alpha=16.0).to("cuda")
        x = torch.randn(4, DIM, device="cuda", dtype=torch.float16)
        out = expert(x)
        max_abs = out.abs().max().item()
        assert max_abs == 0.0, (
            f"LoRAExpert output is {max_abs} at init — is lora_B initialized to zero?"
        )

    with check("LoRAMoEFeedForward is identity on the base FFN at init"):
        class FakeFF(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(DIM, HIDDEN_DIM, bias=False)
                self.w2 = nn.Linear(HIDDEN_DIM, DIM, bias=False)
                self.w3 = nn.Linear(DIM, HIDDEN_DIM, bias=False)
            def forward(self, x):
                return self.w2(F.silu(self.w1(x)) * self.w3(x))
        base_ff = FakeFF().to("cuda").half()
        moe = LoRAMoEFeedForward(
            base_ff=base_ff, dim=DIM, num_experts=4, top_k=2, rank=8, alpha=16.0,
        ).to("cuda")
        x = torch.randn(1, 8, DIM, device="cuda", dtype=torch.float16)
        y_base = base_ff(x)
        y_moe = moe(x)
        diff = (y_base.float() - y_moe.float()).abs().max().item()
        assert diff < 1e-3, (
            f"LoRA-MoE output differs from base FFN by {diff} at init — "
            "must be ≈0 (check that lora_B is zero and base_ff is called directly)"
        )

    # ---- full-model structural checks ----

    with check("convert_to_moe(slice) adds exactly the router parameters"):
        model, _ = load_fresh_model()
        total_before = sum(p.numel() for p in model.parameters())
        model, trainable = convert_to_moe(model, num_experts=4, top_k=2, init_mode="slice")
        added = sum(p.numel() for p in model.parameters()) - total_before
        expected = N_LAYERS * DIM * 4  # router only
        assert added == expected, f"slice added {added:,} params, expected {expected:,}"
        tr = sum(p.numel() for p in trainable)
        assert tr == expected, f"trainable {tr:,} != {expected:,} (router only)"
        del model
        torch.cuda.empty_cache()

    with check("convert_to_moe(lora) has exactly the expected parameter count"):
        model, _ = load_fresh_model()
        total_before = sum(p.numel() for p in model.parameters())
        model, _ = convert_to_moe(
            model, num_experts=4, top_k=2, init_mode="lora", lora_rank=8,
        )
        added = sum(p.numel() for p in model.parameters()) - total_before
        expected_router = N_LAYERS * DIM * 4                        # 131,072
        expected_lora = N_LAYERS * 4 * 2 * DIM * 8                  # 2,097,152
        expected = expected_router + expected_lora                  # 2,228,224
        assert added == expected, (
            f"LoRA mode added {added:,} params, expected {expected:,} "
            f"(router {expected_router:,} + lora {expected_lora:,})"
        )
        del model
        torch.cuda.empty_cache()

    with check("LoRA-MoE preserves dense logits at init (zero-init identity)"):
        dense, tokenizer = load_fresh_model()
        ids = torch.tensor(
            [tokenizer.encode("Hello world", bos=True, eos=False)], device="cuda",
        )
        with torch.no_grad():
            dense_logits = dense(ids, start_pos=0)
        dense_pred = dense_logits[0, -1].argmax().item()
        del dense
        torch.cuda.empty_cache()

        lora_model, _ = load_fresh_model()
        lora_model, _ = convert_to_moe(
            lora_model, num_experts=4, top_k=2, init_mode="lora", lora_rank=8,
        )
        with torch.no_grad():
            lora_logits = lora_model(ids, start_pos=0)
        lora_pred = lora_logits[0, -1].argmax().item()
        max_diff = (dense_logits - lora_logits).abs().max().item()
        assert dense_pred == lora_pred, f"argmax mismatch: dense={dense_pred} lora={lora_pred}"
        assert max_diff < 1e-3, (
            f"max logit diff {max_diff:.6f} — LoRA zero-init should give "
            "exact identity with the dense model"
        )
        del lora_model
        torch.cuda.empty_cache()

    with check("LoRA mode freezes the base FFN and keeps LoRA trainable"):
        model, _ = load_fresh_model()
        model, _ = convert_to_moe(
            model, num_experts=4, top_k=2, init_mode="lora", lora_rank=8,
        )
        ff = model.layers[0].feed_forward
        for name, p in ff.base_ff.named_parameters():
            assert not p.requires_grad, f"base_ff.{name} has requires_grad=True (should be frozen)"
        for i, e in enumerate(ff.lora_experts):
            for name, p in e.named_parameters():
                assert p.requires_grad, f"lora_experts[{i}].{name} has requires_grad=False"
        del model
        torch.cuda.empty_cache()

    with check("random-init expert load is roughly balanced (no collapse)"):
        model, tokenizer = load_fresh_model()
        model, _ = convert_to_moe(model, num_experts=4, top_k=2, init_mode="slice")
        prompts = [
            "The quick brown fox jumps over the lazy dog",
            "What is the meaning of life",
        ]
        stats = get_expert_load_stats(model, tokenizer, prompts, device="cuda")
        assert 0 in stats, "no expert load stats for layer 0 (hook may not be firing)"
        c0 = stats[0]
        total = c0.sum().item()
        pcts = (c0.float() / total * 100).tolist()
        assert min(pcts) >= 1, f"an expert collapsed to {min(pcts):.1f}% at init: {pcts}"
        assert max(pcts) <= 80, f"an expert captured {max(pcts):.1f}% at init: {pcts}"
        del model
        torch.cuda.empty_cache()


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("Implementation check")
    print("=" * 70)
    torch.manual_seed(42)

    quant_checks()
    moe_checks()

    print("\n" + "=" * 70)
    print(f"Results: {_results['pass']} PASS, "
          f"{_results['fail']} FAIL, {_results['skip']} SKIP")
    print("=" * 70)


if __name__ == "__main__":
    main()
