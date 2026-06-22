#!/bin/bash
# Auto-push from B card: pull C/D's RESULTS.md + aggregate + push to GitHub.
# Run via cron every 30 min. Idempotent (only commits if data changed).
set -u
REPO=/root/autodl-tmp/repo
LOG=/root/autodl-tmp/auto_push.log

echo "[$(date)] start" >> $LOG
cd $REPO || { echo "[fail] cd $REPO" >> $LOG; exit 1; }

# Pull latest from remote (in case manual push happened)
git pull --ff-only origin main >> $LOG 2>&1 || true

mkdir -p results/ablations/per_card

# Pull RESULTS.md from each card (B is local; C, D via SSH)
cp /root/autodl-tmp/runs/overnight/RESULTS.md results/ablations/per_card/autodl-b_RESULTS.md 2>/dev/null || true
cp /root/autodl-tmp/runs/overnight/STATUS.txt results/ablations/per_card/autodl-b_STATUS.txt 2>/dev/null || true
ssh autodl-c "cat /root/autodl-tmp/runs/overnight/RESULTS.md 2>/dev/null" > results/ablations/per_card/autodl-c_RESULTS.md 2>/dev/null || true
ssh autodl-c "cat /root/autodl-tmp/runs/overnight/STATUS.txt 2>/dev/null" > results/ablations/per_card/autodl-c_STATUS.txt 2>/dev/null || true
ssh autodl-d "cat /root/autodl-tmp/runs/overnight/RESULTS.md 2>/dev/null" > results/ablations/per_card/autodl-d_RESULTS.md 2>/dev/null || true
ssh autodl-d "cat /root/autodl-tmp/runs/overnight/STATUS.txt 2>/dev/null" > results/ablations/per_card/autodl-d_STATUS.txt 2>/dev/null || true

# Also tar per-config results.json (idempotent)
for src in "local /root/autodl-tmp/runs/overnight" "autodl-c /root/autodl-tmp/runs/overnight" "autodl-d /root/autodl-tmp/runs/overnight"; do
  host=$(echo $src | awk '{print $1}')
  dir=$(echo $src | awk '{print $2}')
  if [ "$host" = "local" ]; then
    cd $dir && find . -maxdepth 2 -name results.json 2>/dev/null | head -300 | xargs -r tar cf - 2>/dev/null \
      | tar xf - -C $REPO/results/ablations/ 2>/dev/null
    cd $REPO
  else
    ssh $host "cd $dir && find . -maxdepth 2 -name results.json 2>/dev/null | head -300 | xargs -r tar cf - 2>/dev/null" \
      | tar xf - -C $REPO/results/ablations/ 2>/dev/null
  fi
done

# Aggregate
python3 finalize/aggregate_local.py >> $LOG 2>&1 || true

# Commit + push if changed
git add -A
if git diff --cached --quiet; then
  echo "[$(date)] no changes" >> $LOG
else
  git commit -m "Auto-update results from cards ($(date +'%F %H:%M'))" >> $LOG 2>&1
  git push origin main >> $LOG 2>&1 && echo "[$(date)] pushed" >> $LOG || echo "[$(date)] push failed" >> $LOG
fi
