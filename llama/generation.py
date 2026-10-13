import torch
from torch import nn


class Generation(nn.Module):
    def generate(self, tokenizer, prompts, max_gen_len, temperature, top_p, kv_caching, device):
        prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        bsz = len(prompt_tokens)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        total_len = max_gen_len + max_prompt_len
        tokens = torch.full((bsz, total_len), tokenizer.pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != tokenizer.pad_id

        prev_pos = 0
        for cur_pos in range(min_prompt_len, total_len):
            with torch.no_grad():
                if kv_caching:
                    # Incremental decoding. On the first iteration prev_pos=0 so we
                    # process the whole prompt; afterwards we only feed the single
                    # new token at position prev_pos, and Attention reads prior
                    # keys/values from its pre-allocated cache (see Attention.forward
                    # in model.py, the `if self.kv_caching:` branch).
                    #
                    # Possible extension (orthogonal to weight quantization):
                    # the cache_k / cache_v tensors in model.py grow linearly with
                    # context length and can dominate memory for long prompts.
                    # Try quantizing them to INT8 or INT4 — short contexts are
                    # weight-bound (weight quantization wins), long contexts become
                    # KV-cache-bound (KV-cache quantization wins).
                    logits = self(tokens[:, prev_pos:cur_pos], prev_pos)
                else:
                    # Not implemented: decoding without the KV cache.
                    #
                    # Implement naive decoding without the KV cache — at every step,
                    # reprocess the entire prefix from scratch: pass
                    # ``tokens[:, :cur_pos]`` with ``start_pos=0`` and let Attention
                    # take its ``else`` branch (which uses xk/xv directly, no cache).
                    # Compare runtime and peak memory against the cached path.
                    pass
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == tokenizer.eos_id
            )

            if kv_caching:
                prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]

            # cut to eos tok if any
            if tokenizer.eos_id in toks:
                eos_idx = toks.index(tokenizer.eos_id)
                toks = toks[:eos_idx]

            out_tokens.append(toks)

        return [{"generation": tokenizer.decode(t)} for t in out_tokens]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
