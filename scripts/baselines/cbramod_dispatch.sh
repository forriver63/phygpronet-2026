#!/usr/bin/env bash
set -u
ulimit -n 65536
source /root/miniconda3/etc/profile.d/conda.sh
conda activate torch110
cd /root/autodl-tmp

ROOT=/root/autodl-tmp/runs/baselines_old/cbramod
mkdir -p "$ROOT"
LOGD="$ROOT"
echo "[dispatch] start $(date)" >> "$LOGD/dispatch.log"

for SEED in 42 43 44 45 46 47; do
  OUT="$ROOT/seed${SEED}"
  mkdir -p "$OUT"
  echo "[dispatch] seed=$SEED START $(date)" >> "$LOGD/dispatch.log"
  python -u train_cbramod_old.py \
      --seed $SEED \
      --out_dir "$OUT" \
      --epochs 20 \
      --batch_size 128 \
      --num_workers 6 \
      > "$LOGD/seed${SEED}.log" 2>&1
  RC=$?
  echo "[dispatch] seed=$SEED END rc=$RC $(date)" >> "$LOGD/dispatch.log"
done
echo "[dispatch] all done $(date)" >> "$LOGD/dispatch.log"
