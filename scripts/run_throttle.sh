#!/bin/bash
# GPU power / temperature / clocks / clock-event (throttle) reasons while running
# idle, the 4096^3 GEMM smoke test, and bench.py fp16 / int4 bs=1.
# Per phase X: logs/dmon_X.log (nvidia-smi dmon -s pucvt, 1 s), logs/smi_X.csv
# (query every 500 ms, incl. each clock-event reason), logs/smi_q_X_{start,mid,end}.txt
# (nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE), logs/throttle_X.log (stdout).
cd "$(dirname "$0")/.."
mkdir -p logs
R=clocks_event_reasons
Q="timestamp,power.draw,power.limit,enforced.power.limit,temperature.gpu,temperature.memory,clocks.sm,clocks.mem,clocks.max.sm,utilization.gpu,utilization.memory"
for r in active gpu_idle applications_clocks_setting sw_power_cap hw_slowdown hw_thermal_slowdown hw_power_brake_slowdown sw_thermal_slowdown sync_boost; do Q="$Q,$R.$r"; done
if ! nvidia-smi --query-gpu="$Q" --format=csv -i 0 > /dev/null 2>&1; then Q=${Q//clocks_event_reasons/clocks_throttle_reasons}; fi
nvidia-smi --query-gpu="$Q" --format=csv -i 0 || { echo "query fields not supported"; exit 1; }
QD=PERFORMANCE,CLOCK,POWER,TEMPERATURE

phase() {  # name, mid-snapshot delay (s), command...
  name=$1; mid=$2; shift 2
  echo "== $name $(date +%T)"
  nvidia-smi -q -d $QD > "logs/smi_q_${name}_start.txt"
  nvidia-smi dmon -i 0 -s pucvt -o T -d 1 > "logs/dmon_$name.log" & D=$!
  nvidia-smi --query-gpu="$Q" --format=csv -i 0 -lms 500 > "logs/smi_$name.csv" & S=$!
  (sleep "$mid"; nvidia-smi -q -d $QD > "logs/smi_q_${name}_mid.txt") & M=$!
  sleep 2
  "$@" > "logs/throttle_$name.log" 2>&1
  wait $M
  nvidia-smi -q -d $QD > "logs/smi_q_${name}_end.txt"
  sleep 2
  kill $D $S
  wait $D $S 2> /dev/null
}

phase idle  5  sleep 10
phase gemm  15 python -u gemm_smoke.py --n 4096 --seconds 30
phase fp16  30 python -u bench.py --mode fp16 --bs 1 --iters 100 --csv results/scratch.csv
phase int4  60 python -u bench.py --mode int4 --bs 1 --iters 100 --csv results/scratch.csv
tail -2 logs/throttle_gemm.log
echo "ALL DONE $(date +%T)"
