"""
INT4 Weight-Only Quantization for Llama 3.2-1B

This module implements per-group INT4 quantization for nn.Linear layers.
Each group of `group_size` weights shares a scale and zero_point.
Two INT4 values are packed into a single uint8 for storage efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """
    A quantized linear layer that stores weights in INT4 format
    and dequantizes them to FP16 at inference time.

    This replaces nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups = in_features // group_size

        # Quantized weights: two INT4 values packed into one uint8
        # Shape: (out_features, in_features // 2) because 2 values per byte
        self.register_buffer("packed_weight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8))

        # Per-group scale and zero_point
        # Shape: (out_features, n_groups)
        self.register_buffer("scale", torch.zeros(out_features, self.n_groups, dtype=torch.float16))
        self.register_buffer("zero_point", torch.zeros(out_features, self.n_groups, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @staticmethod
    def quantize_tensor(weight: torch.Tensor, group_size: int = 128):
        """
        Quantize a FP16 weight tensor to INT4 with per-group scale and zero_point.

        Args:
            weight: FP16 tensor of shape (out_features, in_features)
            group_size: number of weights per group (default: 128)

        Returns:
            packed_weight: uint8 tensor of shape (out_features, in_features // 2)
            scale: FP16 tensor of shape (out_features, n_groups)
            zero_point: FP16 tensor of shape (out_features, n_groups)
        """
        out_features, in_features = weight.shape
        assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        n_groups = in_features // group_size

        # Reshape into groups: (out_features, n_groups, group_size)
        w = weight.float().reshape(out_features, n_groups, group_size)

        w_min = w.amin(dim=-1)
        w_max = w.amax(dim=-1)

        scale = (w_max - w_min) / 15
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero_point = torch.round(-w_min / scale).clamp(0, 15)

        w_int4 = torch.round(w / scale.unsqueeze(-1) + zero_point.unsqueeze(-1)).clamp(0, 15).to(torch.uint8)

        packed_weight = (w_int4[..., 0::2] | (w_int4[..., 1::2] << 4)).reshape(out_features, in_features // 2)

        return packed_weight, scale.half(), zero_point.half()

    @staticmethod
    def dequantize_packed(packed_weight, scale, zero_point, group_size):
        """
        Dequantize packed INT4 weights back to FP16.

        Args:
            packed_weight: uint8 tensor of shape (out_features, in_features // 2)
            scale: FP16 tensor of shape (out_features, n_groups)
            zero_point: FP16 tensor of shape (out_features, n_groups)
            group_size: number of weights per group

        Returns:
            weight: FP16 tensor of shape (out_features, in_features)
        """
        out_features = packed_weight.shape[0]
        in_features = packed_weight.shape[1] * 2
        n_groups = in_features // group_size

        w_even = packed_weight & 0x0F
        w_odd = (packed_weight >> 4) & 0x0F
        w_int4 = torch.empty(out_features, in_features, dtype=torch.uint8, device=packed_weight.device)
        w_int4[:, 0::2] = w_even
        w_int4[:, 1::2] = w_odd

        w_int4 = w_int4.float().reshape(out_features, n_groups, group_size)
        w_fp16 = (w_int4 - zero_point.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)

        return w_fp16.reshape(out_features, in_features).half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dequantize weights, then do matmul.

        Args:
            x: input tensor of shape (..., in_features)

        Returns:
            output tensor of shape (..., out_features)
        """
        # Dequantize weights from INT4 to FP16
        weight = self.dequantize_packed(self.packed_weight, self.scale, self.zero_point, self.group_size)

        # Standard linear operation
        output = F.linear(x, weight, self.bias)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128):
        """
        Create a QuantizedLinear from a pretrained nn.Linear layer.

        Args:
            linear: the original nn.Linear module
            group_size: quantization group size

        Returns:
            QuantizedLinear module with quantized weights
        """
        has_bias = linear.bias is not None
        ql = cls(linear.in_features, linear.out_features, group_size=group_size, bias=has_bias)

        # Quantize the weights
        packed, scale, zp = cls.quantize_tensor(linear.weight.data, group_size)
        ql.packed_weight.copy_(packed)
        ql.scale.copy_(scale.half())
        ql.zero_point.copy_(zp.half())

        if has_bias:
            ql.bias.copy_(linear.bias.data.half())

        return ql


def quantize_model(model, group_size=128):
    """
    Replace all nn.Linear layers in the model with QuantizedLinear.

    Args:
        model: the Llama model
        group_size: quantization group size

    Returns:
        The model with quantized linear layers (modified in-place)
    """
    linear_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    for name, module in linear_layers:
        if name == "":
            return QuantizedLinear.from_linear(module, group_size).to(module.weight.device)

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, QuantizedLinear.from_linear(module, group_size).to(module.weight.device))

    return model


def print_model_size(model):
    """Print the model size in MB."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_mb = (param_bytes + buffer_bytes) / (1024 ** 2)
    print(f"Model size: {total_mb:.2f} MB")
    print(f"  Parameters: {param_bytes / (1024**2):.2f} MB")
    print(f"  Buffers: {buffer_bytes / (1024**2):.2f} MB")
    return total_mb
