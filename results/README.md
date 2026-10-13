# Raw results (A100-PCIE-40GB, torch 2.8.0+cu128)

- `bench.csv` — every `bench.py` run, one row per (mode, bs).
- `logs/` — stdout of each experiment; see `scripts/run_orig.sh` and `scripts/run_sweep.sh` for the commands.

Notes:
- `benchmark_orig_rerun_1014.log` is a second run of `run_benchmark.py` (2026-10-14 02:31). The first run (2026-10-13) was overwritten; its numbers are in EXPERIMENTS.md E0 and agree within ~3%.
- `train_slice_interrupted_1014.log` is a rerun of `train_moe.py --init-mode slice` that was interrupted (Ctrl+C) right after setup; it still contains the dense / pre-training generation samples. The completed 2026-10-13 run's losses are in EXPERIMENTS.md E8 and match epochs 1–2 of `train_slice_ep6.log` exactly.
- nsys reports (`*.nsys-rep`) and memory snapshots (`*.pickle`) are not committed (binary, tens of MB).
