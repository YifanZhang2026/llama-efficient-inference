#!/bin/bash
# nsys captures of the steady state (and one full-process run), then MoE training sweeps.
cd "$(dirname "$0")/.."
mkdir -p prof logs checkpoints
NS="nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true"
CAP="--capture-range=cudaProfilerApi --capture-range-end=stop"

$NS $CAP -o prof/fp16_bs1      python bench.py --mode fp16     --bs 1 --profile > logs/prof_fp16.log 2>&1
$NS $CAP -o prof/int4_bs1      python bench.py --mode int4     --bs 1 --profile > logs/prof_int4.log 2>&1
$NS $CAP -o prof/moelora_bs1   python bench.py --mode moe-lora --bs 1 --profile > logs/prof_moelora.log 2>&1
$NS      -o prof/fp16_bs1_full python bench.py --mode fp16     --bs 1 --iters 2 --csv results/scratch.csv > logs/prof_fp16_full.log 2>&1

# train_moe.py and eval_moe.py both use checkpoints/moe_finetuned.pt, so each run
# is train -> eval -> copy, strictly in sequence. 200 train samples keeps the
# eval split Alpaca[200:300] disjoint.
sweep() {  # name, init_mode, extra train args...
  name=$1; mode=$2; shift 2
  echo "== $name $(date +%T)"
  python -u train_moe.py --init-mode "$mode" "$@" > "logs/train_$name.log" 2>&1 \
  && python -u eval_moe.py --init-mode "$mode" > "logs/eval_$name.log" 2>&1 \
  && cp checkpoints/moe_finetuned.pt "checkpoints/moe_$name.pt" \
  || echo "!! $name failed"
}
sweep slice_ep6      slice --epochs 6
sweep slice_unfreeze slice --unfreeze-experts
sweep lora_lr1e-3    lora  --lr-expert 1e-3
sweep lora_ep6       lora  --epochs 6
echo "ALL DONE $(date +%T)"
