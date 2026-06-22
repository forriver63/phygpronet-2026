#!/bin/bash
set -u
PY=/root/miniconda3/bin/python
SCR=/root/autodl-tmp/train_phyg_v2.py
BASE=/root/autodl-tmp/runs/overnight
LOG=$BASE/exp6_B.log
echo "[exp6_B start] $(date)" >> $LOG

# wait until run_overnight_B.sh exits AND GPU free
while pgrep -f run_overnight_B.sh >/dev/null; do
  sleep 60
done
echo "[exp6_B] run_overnight_B done @ $(date)" >> $LOG
while :; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free_mb" -ge 10000 ]; then break; fi
  sleep 30
done
echo "[exp6_B] gpu free=${free_mb}MiB, starting @ $(date)" >> $LOG

run() {
  local name=$1 ptk=$2 seed=$3
  local out=$BASE/${name}_seed${seed}
  if [ -f $out/results.json ]; then echo "[skip] $name s$seed"; return; fi
  mkdir -p $out
  echo "[run] $name s$seed @ $(date)" >> $LOG
  $PY $SCR --packed_dir /root/autodl-tmp/data/sleepedf_memmap_v2_mech \
    --seed $seed --epochs 5 --K1 96 --tau 3.5 \
    --mech_w 0.2 --pg_topk $ptk --use_pg --num_workers 4 \
    --out_dir $out >> $LOG 2>&1
  echo "[done] $name s$seed exit=$? @ $(date)" >> $LOG
}
for s in 42 43 44; do
  run exp6_topk6   6  $s
  run exp6_topk8   8  $s
  run exp6_topk16  16 $s
  run exp6_topk24  24 $s
done
echo "[exp6_B end] $(date)" >> $LOG
