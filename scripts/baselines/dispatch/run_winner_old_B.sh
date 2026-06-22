#!/bin/bash
set -u
PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/code/proroga/scripts/6_train_inception_tree_protoL1_mech.py
BASE=/root/autodl-tmp/runs/exp1_winner_old
LOG=$BASE/dispatch_B.log
mkdir -p $BASE
echo "[B start] $(date)" >> $LOG
for seed in 45 46 47; do
  out=$BASE/seed${seed}
  if [ -f $out/runs/inception_tree_protoL1_mech_memmap_pg_seed${seed}.pt ]; then echo "[skip] s$seed"; continue; fi
  mkdir -p $out
  cd $out
  echo "[B run] seed=$seed @ $(date)" >> $LOG
  $PY $SCRIPT \
    --packed_dir /root/autodl-tmp/data/sleepedf_memmap_v2_mech \
    --seed $seed --epochs 5 --K1 96 --tau 3.5 --mech_w 0.2 \
    --use_pg --num_workers 4 >> $LOG 2>&1
  echo "[B done] seed=$seed exit=$? @ $(date)" >> $LOG
done
echo "[B end] $(date)" >> $LOG
