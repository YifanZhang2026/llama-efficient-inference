"""
Convert a compressed-tensors W4A16_ASYM checkpoint (GPTQ / AWQ from quant_llmc.py)
into this repo's INT4 format, so the same weights run through QuantizedLinear (v1-v4)
and our eval scripts. Runs in venv-quant (needs only torch + safetensors + transformers).

compressed-tensors "pack-quantized":
  weight_packed      int32 [N, K/8]   8 nibbles per int32 along K, element 0 in bits 0-3
  weight_zero_point  int32 [N/8, G]   8 nibbles per int32 along N
  weight_scale       fp16  [N, G]
  nibble = signed int4 + 8, and W = (q_signed - zp_signed) * scale
Ours (llama/quantize.py): q, zp in [0, 15] with W = (q - zp) * scale, so q = nibble and
zp = nibble directly (the +8 cancels). Packed 2 per uint8, low nibble = even k.

HF -> Meta (llama/model.py's checkpoint layout):
  names: model.layers.i.self_attn.q_proj -> layers.i.attention.wq, mlp.gate_proj -> w1,
         up_proj -> w3, down_proj -> w2, norms, embed_tokens -> tok_embeddings
  rows of q_proj / k_proj: HF permutes them for its rotate_half RoPE; undo with
         w.view(heads, 2, dim/heads/2, K).transpose(1, 2).reshape(dim, K)  (also scale / zp)
lm_head is not quantized in these checkpoints and is tied to the embedding, so
output.weight = tok_embeddings.weight (FP16).

Writes
  <out>/<name>_int4.pt   {meta linear name: (packed uint8, scale fp16, zp fp16)} + FP16 rest
  <out>/<name>_fake.pth  Meta-key fp16 state_dict with every Linear = dequantized INT4
                         (input for eval_ppl-style scripts and eval_tasks.py --state-dict)
Check: our dequant (HF row order) vs transformers' own decompressed weights, per layer.

Usage (venv-quant):
    /root/autodl-tmp/venv-quant/bin/python convert_ct.py --name gptq --ckpt /root/autodl-tmp/llmc/gptq_w4a16_asym
"""
import argparse
import os

import torch
from safetensors import safe_open

N_HEADS, N_KV, DIM = 32, 8, 2048
MAP = {"self_attn.q_proj": "attention.wq", "self_attn.k_proj": "attention.wk", "self_attn.v_proj": "attention.wv",
       "self_attn.o_proj": "attention.wo", "mlp.gate_proj": "feed_forward.w1", "mlp.up_proj": "feed_forward.w3",
       "mlp.down_proj": "feed_forward.w2"}


def unpack_int32(x, axis):
    """int32 tensor with 8 nibbles per element along `axis` -> uint8 nibbles, element 0 in the low bits."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=x.device)
    if axis == 1:
        n = (x.unsqueeze(-1) >> shifts) & 0xF                 # [N, K/8, 8]
        return n.reshape(x.shape[0], -1).to(torch.uint8)
    n = (x.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF       # [N/8, 8, G]
    return n.reshape(-1, x.shape[1]).to(torch.uint8)


def unpermute_rows(t, heads):
    rows = t.shape[0]
    return t.view(heads, 2, rows // heads // 2, *t.shape[1:]).transpose(1, 2).reshape(rows, *t.shape[1:])


def dequant(q, zp, scale, g=128):
    """Exactly QuantizedLinear.dequantize_packed (v1): fp32 (q - zp) * scale, then fp16."""
    n, k = q.shape
    w = (q.float().reshape(n, k // g, g) - zp.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)
    return w.reshape(n, k).half()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="/root/autodl-tmp/fakequant")
    p.add_argument("--no-check", action="store_true")
    a = p.parse_args()
    f = safe_open(os.path.join(a.ckpt, "model.safetensors"), "pt", device="cuda")
    keys = set(f.keys())
    int4, fake = {}, {}
    hf_q = {}
    for i in range(16):
        for hf, meta in MAP.items():
            pre = f"model.layers.{i}.{hf}."
            packed = f.get_tensor(pre + "weight_packed")
            scale = f.get_tensor(pre + "weight_scale")
            zpp = f.get_tensor(pre + "weight_zero_point")
            n, k = (int(v) for v in f.get_tensor(pre + "weight_shape").tolist())
            q = unpack_int32(packed, 1)[:, :k]
            zp = unpack_int32(zpp, 0)[:n]
            assert q.shape == (n, k) and zp.shape == scale.shape, (pre, q.shape, zp.shape, scale.shape)
            hf_q[f"model.layers.{i}.{hf}"] = (q, zp, scale)
            if hf in ("self_attn.q_proj", "self_attn.k_proj"):
                heads = N_HEADS if hf == "self_attn.q_proj" else N_KV
                q, zp, scale = unpermute_rows(q, heads), unpermute_rows(zp, heads), unpermute_rows(scale, heads)
            name = f"layers.{i}.{meta}"
            packed8 = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
            int4[name] = (packed8.cpu(), scale.half().cpu(), zp.half().cpu())
            fake[name + ".weight"] = dequant(q, zp, scale).cpu()
        fake[f"layers.{i}.attention_norm.weight"] = f.get_tensor(f"model.layers.{i}.input_layernorm.weight").cpu()
        fake[f"layers.{i}.ffn_norm.weight"] = f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight").cpu()
    fake["norm.weight"] = f.get_tensor("model.norm.weight").cpu()
    emb = f.get_tensor("model.embed_tokens.weight").cpu()
    fake["tok_embeddings.weight"] = emb
    fake["output.weight"] = f.get_tensor("lm_head.weight").cpu() if "lm_head.weight" in keys else emb
    print(f"converted {len(int4)} Linear; lm_head {'from lm_head.weight' if 'lm_head.weight' in keys else 'tied to embed_tokens'} (FP16)")

    if not a.no_check:
        from transformers import AutoModelForCausalLM, CompressedTensorsConfig
        m = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.float16, device_map="cuda",
                                                 quantization_config=CompressedTensorsConfig(run_compressed=False))
        worst = 0.0
        for name, mod in m.named_modules():
            if name in hf_q:
                q, zp, scale = hf_q[name]
                d = (dequant(q, zp, scale).float() - mod.weight.data.float()).abs().max().item()
                worst = max(worst, d)
        print(f"check vs transformers decompression: max |ours - HF| over 112 Linear = {worst:.3e}")
        del m

    os.makedirs(a.out, exist_ok=True)
    torch.save({"int4": int4, "group_size": 128, "source": a.ckpt}, os.path.join(a.out, f"{a.name}_int4.pt"))
    torch.save(fake, os.path.join(a.out, f"{a.name}_fake.pth"))
    print(f"saved {a.out}/{a.name}_int4.pt and {a.name}_fake.pth")


if __name__ == "__main__":
    main()
