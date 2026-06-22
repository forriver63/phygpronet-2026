#!/bin/bash
set -u
PY=/root/miniconda3/bin/python
SCR=/root/autodl-tmp/train_phyg_v2.py
BASE=/root/autodl-tmp/runs/overnight
LOG=$BASE/master_B.log
mkdir -p $BASE
echo "[B start] $(date)" >> $LOG

run() {
  local name=$1 ep=$2 mw=$3 wu=$4 hcp=$5 sww=$6 spw=$7 ptk=$8 seed=$9
  local out=$BASE/${name}_seed${seed}
  if [ -f $out/results.json ]; then echo "[B skip] $name s$seed"; return; fi
  mkdir -p $out
  echo "[B run] $name s$seed @ $(date)" >> $LOG
  $PY $SCR --packed_dir /root/autodl-tmp/data/sleepedf_memmap_v2_mech \
    --seed $seed --epochs $ep --K1 96 --tau 3.5 \
    --mech_w $mw --mech_warmup_epoch $wu --mech_high_conf_pct $hcp \
    --slowwave_w $sww --spindle_w $spw --pg_topk $ptk \
    --use_pg --num_workers 4 --out_dir $out >> $LOG 2>&1
  echo "[B done] $name s$seed exit=$? @ $(date)" >> $LOG
}

# B works backwards from Exp 6 (least likely to overlap with A)
for s in 44 43 42; do
  run exp6_topk24  5 0.2 0 1.0 0 0 24 $s
  run exp6_topk16  5 0.2 0 1.0 0 0 16 $s
  run exp6_topk8   5 0.2 0 1.0 0 0 8 $s
  run exp6_topk6   5 0.2 0 1.0 0 0 6 $s
done
for s in 44 43 42; do
  run exp4_sw15sp05  5 0.0 0 1.0 0.15 0.05 12 $s
  run exp4_sw20sp10  5 0.0 0 1.0 0.20 0.10 12 $s
  run exp4_sw20sp05  5 0.0 0 1.0 0.20 0.05 12 $s
done
for s in 44 43; do
  run exp3_hc40  5 0.2 0 0.4 0 0 12 $s
  run exp3_hc30  5 0.2 0 0.3 0 0 12 $s
  run exp3_hc20  5 0.2 0 0.2 0 0 12 $s
done
echo "[B end] $(date)" >> $LOG
