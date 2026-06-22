#!/bin/bash
set -u
PY=/root/miniconda3/bin/python
SCR=/root/autodl-tmp/eval_exp1_perckpt.py
SPLIT=/root/autodl-tmp/splits/sleepedf_subject_5fold_seed42.json
BASE=/root/autodl-tmp/runs/overnight
LOG=$BASE/eval_zs.log
MARK=$1   # half A or B; A gets 1st half, B gets 2nd
echo "[eval $MARK start] $(date)" >> $LOG

all=( $(ls $BASE/*/best_model.pt 2>/dev/null | sort) )
nh=$((${#all[@]} / 2))
if [ "$MARK" = "A" ]; then
  sel=("${all[@]:0:$nh}")
else
  sel=("${all[@]:$nh}")
fi
for f in "${sel[@]}"; do
  d=$(dirname $f); n=$(basename $d)
  if [ -f $d/zs_eval.csv ]; then echo "[skip $MARK] $n"; continue; fi
  mkdir -p $d/_single
  cp $f $d/_single/epoch_99.pt
  echo "[eval $MARK] $n @ $(date)" >> $LOG
  $PY $SCR --fold_dir $d/_single --fold_id 0 --split_json $SPLIT --out_csv $d/zs_eval.csv >> $LOG 2>&1
  rm -rf $d/_single
done
echo "[eval $MARK end] $(date)" >> $LOG
