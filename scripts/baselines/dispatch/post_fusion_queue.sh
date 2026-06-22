#!/bin/bash
# Wait for fusionA to finish on both machines, then dispatch follow-up evals.
set -u
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/runs/overnight/post_fusion.log
echo "[post_fusion start] $(date)" >> $LOG

# 1) Wait for fusionA training to fully complete (5 seeds on this machine)
echo "[wait] for fusionA to finish on this machine ..." >> $LOG
while pgrep -f run_fusionA >/dev/null || pgrep -f 'train_phyg_v2.*fusionA' >/dev/null; do
  sleep 60
done
echo "[wait] fusionA done @ $(date)" >> $LOG

# 2) Evaluate SHHS per-class for fusionA seeds present here
echo "[eval] fusionA SHHS per-class @ $(date)" >> $LOG
for s in 42 43 44 45 46; do
  d=/root/autodl-tmp/runs/overnight/fusionA_seed${s}
  if [ ! -f $d/best_model.pt ]; then continue; fi
  if [ -f $d/shhs_perclass.txt ]; then continue; fi
  echo "  -- s$s --" >> $LOG
  $PY /root/autodl-tmp/eval_shhs_perclass.py --ckpt $d/best_model.pt > $d/shhs_perclass.txt 2>&1
done
echo "[eval] fusionA done @ $(date)" >> $LOG

# 3) Exp 8: temperature calibration on winner (exp3_hc20) seeds present here
echo "[exp8] temperature calibration @ $(date)" >> $LOG
for s in 42 43 44 45 46; do
  d=/root/autodl-tmp/runs/overnight/exp3_hc20_seed${s}
  if [ ! -f $d/best_model.pt ]; then continue; fi
  if [ -f $d/temp_cal.txt ]; then continue; fi
  $PY /root/autodl-tmp/eval_temp_cal.py --ckpt $d/best_model.pt > $d/temp_cal.txt 2>&1 || true
done
echo "[exp8] done @ $(date)" >> $LOG
echo "[post_fusion end] $(date)" >> $LOG
