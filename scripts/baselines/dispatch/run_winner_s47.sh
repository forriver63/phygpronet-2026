#!/bin/bash
set -u
PY=/root/miniconda3/bin/python
SCR=/root/autodl-tmp/train_phyg_v2.py
out=/root/autodl-tmp/runs/overnight/exp3_hc20_seed47
mkdir -p $out
LOG=$out/run.log
echo "[start] $(date)" >> $LOG
$PY $SCR --packed_dir /root/autodl-tmp/data/sleepedf_memmap_v2_mech \
  --seed 47 --epochs 5 --K1 96 --tau 3.5 \
  --mech_w 0.2 --mech_high_conf_pct 0.2 \
  --pg_topk 12 --use_pg --num_workers 4 --out_dir $out >> $LOG 2>&1
echo "[done] exit=$? @ $(date)" >> $LOG
echo "[shhs eval]" >> $LOG
$PY /root/autodl-tmp/eval_shhs_perclass.py --ckpt $out/best_model.pt >> $LOG 2>&1
echo "[mass eval]" >> $LOG
$PY /root/autodl-tmp/eval_mass_isruc.py --ckpt $out/best_model.pt --target mass >> $LOG 2>&1
echo "[isruc eval]" >> $LOG
$PY /root/autodl-tmp/eval_mass_isruc.py --ckpt $out/best_model.pt --target isruc >> $LOG 2>&1
echo "[all done] $(date)" >> $LOG
