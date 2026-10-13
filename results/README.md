# Raw results (A100-PCIE-40GB, torch 2.8.0+cu128)

- `bench.csv` — every `bench.py` run, one row per (mode, bs).
- `logs/` — stdout of each experiment; see `scripts/run_orig.sh`, `scripts/run_sweep.sh` and `scripts/run_lora_seeds.sh` for the commands.
- `ppl.csv` — `eval_ppl.py`: FP16 vs INT4 (g32/64/128) PPL, acc, agreement and KL on Alpaca[200:300] and WikiText-2 (EXPERIMENTS.md E1b).
- `slice_scale.csv` — `slice_scale.py`: untrained slice-MoE vs dense FFN norm ratio / cosine per layer (E7b).
- `update_rounding.csv` — `update_rounding.py`: per-step fraction of SGD updates lost to rounding (E8c). Rows with `group=lora` come from the first run, which pooled LoRA A and B; use the `lora_A` / `lora_B` rows instead.

Notes:
- `benchmark_orig_rerun_1014.log` is a second run of `run_benchmark.py` (2026-10-14 02:31). The first run (2026-10-13) was overwritten; its numbers are in EXPERIMENTS.md E0 and agree within ~3%.
- `train_slice_interrupted_1014.log` is a rerun of `train_moe.py --init-mode slice` that was interrupted (Ctrl+C) right after setup; it still contains the dense / pre-training generation samples. The completed 2026-10-13 run's losses are in EXPERIMENTS.md E8 and match epochs 1–2 of `train_slice_ep6.log` exactly.
- nsys reports (`*.nsys-rep`) and memory snapshots (`*.pickle`) are not committed (binary, tens of MB).
- `run_lora_seeds_1014.log` ran seeds 1–3 (the version of the script before seed 42 became a default); `run_lora_seeds_42_1014.log` is seed 42, started 4 min later and run concurrently. Its final summary lists all four seeds' eval logs; the first one's lists seed 42 before it finished.
- `logs/benchmark_orig.log` / `logs/train_slice.log` recovered from an old server-side commit turned out to be byte-identical to `benchmark_orig_rerun_1014.log` / `train_slice_interrupted_1014.log`, so the 2026-10-13 originals are not recoverable.
