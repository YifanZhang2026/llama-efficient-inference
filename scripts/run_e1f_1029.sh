#!/bin/bash
# E1f-A (2026-10-29): mixed precision + per-group clip search, no calibration data.
cd "$(dirname "$0")/.."
mkdir -p logs
python -u quant_improve.py --save clip_g128 clip_g128+w2_g32 clip_g128+w2_g32+block1_fp16 > logs/quant_improve_1029.log 2>&1
grep -v Warning logs/quant_improve_1029.log | grep -E "GPU:|built|FP16 size|^\[|saved|appended"
tar czf artifacts_1029e1f.tgz logs/quant_improve_1029.log results/quant_improve.csv
ls -la artifacts_1029e1f.tgz
