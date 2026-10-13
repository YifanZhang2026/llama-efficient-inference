"""
Download Alpaca training data and MT-bench evaluation prompts.

Writes alpaca_500.json and mt_bench_prompts.json. Requires the `datasets` package:
    pip install datasets

Usage:
    python prepare_data.py                         # default: ./data/
    python prepare_data.py --output-dir /tmp/data  # custom location
"""
import argparse
import json
import os


def download_alpaca(output_path, n_samples=500):
    """Download and save Alpaca dataset samples."""
    from datasets import load_dataset
    print("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    samples = []
    for item in ds:
        if item["output"] and len(item["output"]) > 10 and len(item["output"]) < 500:
            text = ""
            if item["instruction"]:
                text += item["instruction"]
            if item["input"]:
                text += " " + item["input"]
            text += " " + item["output"]
            samples.append({"text": text, "instruction": item["instruction"],
                            "input": item.get("input", ""), "output": item["output"]})
        if len(samples) >= n_samples:
            break

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} Alpaca samples to {output_path}")


def create_mt_bench_prompts(output_path):
    """Create a subset of MT-bench style prompts for evaluation."""
    prompts = [
        # Writing
        "Compose a short poem about the beauty of mathematics.",
        "Write a persuasive paragraph arguing that reading books is better than watching TV.",
        "Draft a professional email declining a job offer politely.",
        # Reasoning
        "If a train travels at 60 mph for 2.5 hours, how far does it go? Show your reasoning.",
        "What are three key differences between classical and operant conditioning?",
        "Explain why the sky appears blue during the day but red at sunset.",
        # Coding
        "Write a Python function that checks if a string is a palindrome.",
        "Explain the difference between a stack and a queue data structure.",
        # Knowledge
        "What were the main causes of World War I? List at least three.",
        "Describe the process of photosynthesis in simple terms.",
        "What is the significance of the Turing test in artificial intelligence?",
        "Explain how a neural network learns during backpropagation.",
        # Math
        "Solve: If 3x + 7 = 22, what is x?",
        "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?",
        # General
        "What are the pros and cons of remote work?",
        "Summarize the main ideas behind stoic philosophy in three sentences.",
        "If you could have dinner with any historical figure, who would it be and why?",
        "Explain the concept of supply and demand to a 10-year-old.",
        "What makes a good leader? Give three qualities with brief explanations.",
        "Describe how climate change affects ocean ecosystems.",
    ]
    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} MT-bench prompts to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output-dir", default="./data",
                        help="Directory to save the two JSON files (default: ./data)")
    parser.add_argument("--n-alpaca", type=int, default=500,
                        help="Number of Alpaca samples to keep (default: 500)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    download_alpaca(os.path.join(args.output_dir, "alpaca_500.json"), args.n_alpaca)
    create_mt_bench_prompts(os.path.join(args.output_dir, "mt_bench_prompts.json"))
