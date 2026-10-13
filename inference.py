import os

import torch

from llama.config import CKPT_DIR
from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama

def inference():
    """Smoke test: load the dense FP16 model and print a few sample generations."""
    torch.manual_seed(1)

    checkpoint_dir = CKPT_DIR
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
    model_path = os.path.join(checkpoint_dir, "consolidated.00.pth")

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    torch.set_default_tensor_type(torch.cuda.HalfTensor) # load model in fp16
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    device = "cuda"
    model.to(device)
    
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    model.eval()
    results = model.generate(tokenizer, prompts, max_gen_len=64, temperature=0.6, top_p=0.9, kv_caching=True, device=device)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    
if __name__ == "__main__":
    inference()
