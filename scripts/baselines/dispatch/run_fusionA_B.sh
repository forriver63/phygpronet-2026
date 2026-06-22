#!/bin/bash
set -u
PY=/root/miniconda3/bin/python
SCR=/root/autodl-tmp/train_phyg_v2.py
BASE=/root/autodl-tmp/runs/overnight
LOG=$BASE/fusionA.log
mkdir -p $BASE
echo "[fusionA start] $(date)" >> $LOG
# slowwave_w=0.15, spindle_w=0.10, sp_hcp=0.2 (top20% spindle), sw_hcp=1.0 (full)
for s in 42 43 44 45 46; do
  out=$BASE/fusionA_seed${s}
  if [ -f $out/results.json ]; then echo skip s$s; continue; fi
  mkdir -p $out
  echo "[run] s$s @ $(date)" >> $LOG
  $PY $SCR --packed_dir /root/autodl-tmp/data/sleepedf_memmap_v2_mech \
    --seed $s --epochs 5 --K1 96 --tau 3.5 \
    --mech_w 0.0 --slowwave_w 0.15 --spindle_w 0.10 \
    --mech_high_conf_pct_sp 0.2 --mech_high_conf_pct_sw 1.0 \
    --pg_topk 12 --use_pg --num_workers 4 \
    --out_dir $out >> $LOG 2>&1
  echo "[done] s$s exit=$? @ $(date)" >> $LOG
done
echo "[fusionA end] $(date)" >> $LOG
