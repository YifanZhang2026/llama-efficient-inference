#!/bin/bash
# EXP_HANDOFF items 6-8 (2026-10-15): dequant HBM traffic, symmetric vs asymmetric
# INT4, then clocks / throttle reasons. Runs strictly in sequence (the throttle
# phase must have the GPU to itself). New log names only; nothing is overwritten.
cd "$(dirname "$0")/.."
mkdir -p logs
echo "== dequant_traffic A+B $(date +%T)"
python -u dequant_traffic.py > logs/dequant_traffic_1015.log 2>&1

echo "== dequant_traffic C (ncu) $(date +%T)"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
if [ -n "$NCU" ]; then
  echo "ncu: $NCU" > logs/ncu_dequant_1015.log
  "$NCU" --profile-from-start off --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
    --csv --log-file logs/ncu_dequant_1015.csv python -u dequant_traffic.py --ncu >> logs/ncu_dequant_1015.log 2>&1
  echo "ncu exit $?" >> logs/ncu_dequant_1015.log
else
  echo "ncu not found" > logs/ncu_dequant_1015.log
fi
tail -3 logs/ncu_dequant_1015.log

echo "== quant_sym $(date +%T)"
python -u quant_sym.py > logs/quant_sym_1015.log 2>&1

bash scripts/run_throttle.sh
echo "ALL DONE $(date +%T)"
