#!/bin/bash
set -u
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/runs/exp1_saveall/dispatch_B.log
mkdir -p /root/autodl-tmp/runs/exp1_saveall
echo "[B start] $(date)" >> $LOG
for f in 3 4; do
  out=/root/autodl-tmp/runs/exp1_saveall/fold$f
  mkdir -p $out
  echo "[B run fold$f] $(date)" >> $LOG
  $PY /root/autodl-tmp/train_phyg_unified_saveall.py \
    --split_json /root/autodl-tmp/splits/sleepedf_subject_5fold_seed42.json \
    --fold_id $f --out_dir $out \
    --K1 192 --tau 3.5 --mech_w 0.1 --epochs 50 \
    --num_workers 4 >> $LOG 2>&1
  echo "[B done fold$f exit=$?] $(date)" >> $LOG
done
echo "[B end] $(date)" >> $LOG
