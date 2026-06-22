#!/bin/bash
set -u
PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/train_phyg_unified.py
SPLIT=/root/autodl-tmp/splits/sleepedf_subject_5fold_seed42.json
BASE=/root/autodl-tmp/runs/exp1_hpsweep
LOG=$BASE/dispatch_B.log
mkdir -p $BASE
echo "[B start] $(date)" >> $LOG

declare -a configs=(
  "e15_mw01_k192_lscw 15 0.1  192  sqrt_clipped   0.05"
  "e10_mw02_k192      10 0.2  192  none           0.0"
  "e7_mw02_k96        7  0.2  96   none           0.0"
)
for seed in 42 43 44; do
  for cfg in "${configs[@]}"; do
    set -- $cfg
    name=$1; ep=$2; mw=$3; k1=$4; cw=$5; ls=$6
    out=$BASE/${name}_seed${seed}
    if [ -f $out/results.json ]; then echo "[skip] $name s$seed"; continue; fi
    mkdir -p $out
    echo "[B run] $name seed=$seed @ $(date)" >> $LOG
    $PY $SCRIPT \
      --split_json $SPLIT --fold_id 0 --out_dir $out \
      --seed $seed --epochs $ep --K1 $k1 --tau 3.5 --mech_w $mw \
      --class_weights $cw --label_smoothing $ls \
      --num_workers 4 >> $LOG 2>&1
    echo "[B done] $name seed=$seed exit=$? @ $(date)" >> $LOG
  done
done
echo "[B end] $(date)" >> $LOG
