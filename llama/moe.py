"""
Mixture-of-Experts (MoE) FeedForward layer for Llama 3.2-1B

This module implements two MoE variants:

1. ``slice`` init — split the pretrained dense FFN into N smaller expert FFNs,
   each with hidden_dim / N.  A learned router selects top-K experts per token.

2. ``lora`` init — keep the original dense FFN **frozen** and attach N lightweight
   LoRA adapters as "experts".  The router selects which adapters to activate.
   This is far more parameter-efficient than copying the full FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# Part A: Slice-based MoE
# ===================================================================

class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts FeedForward layer (slice variant).

    Each expert has the same SwiGLU structure (w1, w2, w3) as the original FFN,
    but with hidden_dim / num_experts.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        """
        Args:
            dim: input/output dimension (model embedding dim)
            hidden_dim: hidden dimension of the original dense FFN
            num_experts: number of experts (N)
            top_k: number of experts activated per token (K)
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden_dim = hidden_dim // num_experts

        # Router: maps input to expert scores
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Experts: each is a small SwiGLU FFN (same structure as Llama FeedForward)
        self.expert_w1 = nn.ModuleList([nn.Linear(dim, self.expert_hidden_dim, bias=False) for _ in range(num_experts)])
        self.expert_w2 = nn.ModuleList([nn.Linear(self.expert_hidden_dim, dim, bias=False) for _ in range(num_experts)])
        self.expert_w3 = nn.ModuleList([nn.Linear(dim, self.expert_hidden_dim, bias=False) for _ in range(num_experts)])

        # For tracking expert activations (used in analysis)
        self._last_routing_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, dim)

        Returns:
            output tensor of shape (batch_size, seq_len, dim)
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]

        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        self._last_routing_indices = top_k_indices.detach()

        output_flat = torch.zeros_like(x_flat)
        for token_idx in range(num_tokens):
            token_input = x_flat[token_idx : token_idx + 1]
            for k in range(self.top_k):
                expert_idx = int(top_k_indices[token_idx, k].item())
                weight = top_k_weights[token_idx, k].to(x_flat.dtype)
                expert_output = self._expert_forward(expert_idx, token_input).squeeze(0)
                output_flat[token_idx] = output_flat[token_idx] + weight * expert_output

        return output_flat.reshape(original_shape)

    def _expert_forward(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Run a single expert's SwiGLU FFN."""
        return self.expert_w2[expert_idx](
            F.silu(self.expert_w1[expert_idx](x)) * self.expert_w3[expert_idx](x)
        )


# ===================================================================
# Part B: LoRA-based MoE
# ===================================================================
#
# Motivation: slice mode (Part A) splits the pretrained FFN into N smaller
# experts. With top_k < N active experts per token and only a few hundred
# fine-tuning samples, each expert loses ~half of its original capacity and
# the router hasn't learned to specialize — quality drops sharply. LoRA MoE
# solves this by keeping the full FFN intact and adding tiny "diff" experts
# on top.
#
# Recall from LoRA (Hu et al., 2021):
#   ΔW = B @ A,  where A ∈ R^{r×d}, B ∈ R^{d×r}, r << d
#
# Here each expert is one such (A, B) pair added to the frozen FFN output:
#   output = base_ffn(x) + Σ_k  weight_k · lora_expert_k(x)
#
# At initialization, B is zero → every expert contributes zero →
# the converted model behaves identically to the original dense model.

class LoRAExpert(nn.Module):
    """
    A single LoRA adapter used as a lightweight expert.

    Architecture:  x → A (down-project to rank r) → B (up-project back to dim)
    """

    def __init__(self, dim: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(dim, rank, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, dim, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.lora_B(self.lora_A(x.to(torch.float32))) * self.scaling
        return output.to(x.dtype)


class LoRAMoEFeedForward(nn.Module):
    """
    Frozen shared FFN + routed LoRA experts.

    The base FFN is the original pretrained FeedForward — kept frozen so it
    adds zero extra memory.  Each LoRA expert adds only 2 * dim * rank
    parameters, which is tiny compared to a full FFN copy.
    """

    def __init__(self, base_ff: nn.Module, dim: int, num_experts: int = 4,
                 top_k: int = 2, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank

        self.base_ff = base_ff
        for param in self.base_ff.parameters():
            param.requires_grad = False
        self.router = nn.Linear(dim, num_experts, bias=False, dtype=torch.float32)
        self.lora_experts = nn.ModuleList(
            [LoRAExpert(dim, rank=rank, alpha=alpha) for _ in range(num_experts)]
        )
        self._last_routing_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]

        with torch.no_grad():
            base_output_flat = self.base_ff(x_flat)

        router_logits = self.router(x_flat.to(torch.float32))
        routing_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        self._last_routing_indices = top_k_indices.detach()

        lora_output_flat = torch.zeros_like(x_flat)
        for token_idx in range(num_tokens):
            token_input = x_flat[token_idx : token_idx + 1]
            for k in range(self.top_k):
                expert_idx = int(top_k_indices[token_idx, k].item())
                weight = top_k_weights[token_idx, k].to(x_flat.dtype)
                expert_output = self.lora_experts[expert_idx](token_input).squeeze(0)
                lora_output_flat[token_idx] = lora_output_flat[token_idx] + weight * expert_output

        return (base_output_flat + lora_output_flat).reshape(original_shape)


# ===================================================================
# Model conversion
# ===================================================================

def convert_to_moe(model, num_experts=4, top_k=2, init_mode="slice",
                   lora_rank=8, lora_alpha=16.0):
    """
    Replace all FeedForward layers in the model with MoE variants.

    Args:
        model: the Llama model
        num_experts: number of experts per layer
        top_k: number of active experts per token
        init_mode: ``"slice"`` or ``"lora"``
        lora_rank: rank for LoRA experts (only used when init_mode="lora")
        lora_alpha: LoRA scaling factor; effective scale = alpha / rank

    Returns:
        (model, trainable_params): the modified model and a flat list of
        parameters that should be passed to the optimizer.
    """
    trainable_params = []

    for layer in model.layers:
        old_ff = layer.feed_forward
        hidden_dim = old_ff.w1.out_features  # original hidden dim

        if init_mode == "lora":
            moe_ff = LoRAMoEFeedForward(
                base_ff=old_ff,
                dim=old_ff.w1.in_features,
                num_experts=num_experts,
                top_k=top_k,
                rank=lora_rank,
                alpha=lora_alpha,
            ).to(device=old_ff.w1.weight.device)
            layer.feed_forward = moe_ff
            trainable_params.extend(moe_ff.router.parameters())
            for expert in moe_ff.lora_experts:
                trainable_params.extend(expert.parameters())

        elif init_mode == "slice":
            moe_ff = MoEFeedForward(
                dim=old_ff.w1.in_features,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
            ).to(device=old_ff.w1.weight.device, dtype=old_ff.w1.weight.dtype)

            with torch.no_grad():
                for i in range(num_experts):
                    slice_start = i * (hidden_dim // num_experts)
                    slice_end = (i + 1) * (hidden_dim // num_experts)
                    moe_ff.expert_w1[i].weight.copy_(old_ff.w1.weight[slice_start:slice_end, :])
                    moe_ff.expert_w2[i].weight.copy_(old_ff.w2.weight[:, slice_start:slice_end])
                    moe_ff.expert_w3[i].weight.copy_(old_ff.w3.weight[slice_start:slice_end, :])

            layer.feed_forward = moe_ff
            trainable_params.extend(moe_ff.router.parameters())

        else:
            raise ValueError(f"init_mode must be 'slice' or 'lora', got {init_mode!r}")

    return model, trainable_params


# ===================================================================
# Analysis helper
# ===================================================================

def get_expert_load_stats(model, tokenizer, prompts, device="cuda"):
    """
    Run prompts through the model and collect expert activation statistics.

    Returns a dict mapping layer_idx -> tensor of shape (num_experts,)
    with activation counts.
    """
    stats = {}
    for layer_idx, layer in enumerate(model.layers):
        ff = layer.feed_forward
        if hasattr(ff, "num_experts") and hasattr(ff, "_last_routing_indices"):
            stats[layer_idx] = torch.zeros(ff.num_experts, dtype=torch.long)

    was_training = model.training
    model.eval()
    model.to(device)

    with torch.no_grad():
        for prompt in prompts:
            token_ids = tokenizer.encode(prompt, bos=True, eos=False)
            if len(token_ids) == 0:
                continue

            tokens = torch.tensor([token_ids], device=device, dtype=torch.long)
            _ = model(tokens, start_pos=0)

            for layer_idx, layer in enumerate(model.layers):
                if layer_idx not in stats:
                    continue
                ff = layer.feed_forward
                routing_indices = ff._last_routing_indices
                if routing_indices is None:
                    continue
                counts = torch.bincount(
                    routing_indices.reshape(-1),
                    minlength=ff.num_experts,
                ).to(torch.long).cpu()
                stats[layer_idx] += counts

    if was_training:
        model.train()

    return stats
