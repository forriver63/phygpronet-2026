#!/bin/bash
set -u
REPO=/root/autodl-tmp/repo
LOG=/root/autodl-tmp/auto_push.log
echo "[$(date)] start" >> $LOG
cd $REPO || exit 1
git pull --ff-only origin main >> $LOG 2>&1 || true
mkdir -p results/ablations/per_card
cp /root/autodl-tmp/runs/overnight/RESULTS.md results/ablations/per_card/autodl-b_RESULTS.md 2>/dev/null || true
cp /root/autodl-tmp/runs/overnight/STATUS.txt results/ablations/per_card/autodl-b_STATUS.txt 2>/dev/null || true
for h in autodl-c autodl-d autodl-e; do
  ssh $h "cat /root/autodl-tmp/runs/overnight/RESULTS.md 2>/dev/null" > results/ablations/per_card/${h}_RESULTS.md 2>/dev/null || true
  ssh $h "cat /root/autodl-tmp/runs/overnight/STATUS.txt 2>/dev/null" > results/ablations/per_card/${h}_STATUS.txt 2>/dev/null || true
done
# Tar per-config results.json + fewshot json from all 4 cards
for src in "local /root/autodl-tmp/runs/overnight" "autodl-c /root/autodl-tmp/runs/overnight" "autodl-d /root/autodl-tmp/runs/overnight" "autodl-e /root/autodl-tmp/runs/overnight"; do
  host=$(echo $src | awk '{print $1}')
  dir=$(echo $src | awk '{print $2}')
  if [ "$host" = "local" ]; then
    cd $dir && find . -maxdepth 2 -name results.json 2>/dev/null | head -400 | xargs -r tar cf - 2>/dev/null \
      | tar xf - -C $REPO/results/ablations/ 2>/dev/null
    # Fewshot json dirs
    for fs_dir in shhs_fs2 mass_fs isruc_fs; do
      if [ -d $dir/$fs_dir ]; then
        cd $dir && tar cf - $fs_dir/*.json 2>/dev/null | tar xf - -C $REPO/results/fewshot/ 2>/dev/null
      fi
    done
    cd $REPO
  else
    ssh $host "cd $dir && find . -maxdepth 2 -name results.json 2>/dev/null | head -400 | xargs -r tar cf - 2>/dev/null" \
      | tar xf - -C $REPO/results/ablations/ 2>/dev/null
    ssh $host "cd $dir && tar cf - shhs_fs2/*.json mass_fs/*.json isruc_fs/*.json 2>/dev/null" \
      | tar xf - -C $REPO/results/fewshot/ 2>/dev/null
  fi
done
# Normalize fewshot dirs: shhs_fs2 -> shhs (the repo standard)
[ -d $REPO/results/fewshot/shhs_fs2 ] && rsync -a $REPO/results/fewshot/shhs_fs2/ $REPO/results/fewshot/shhs/ && rm -rf $REPO/results/fewshot/shhs_fs2
[ -d $REPO/results/fewshot/mass_fs ] && rsync -a $REPO/results/fewshot/mass_fs/ $REPO/results/fewshot/mass/ && rm -rf $REPO/results/fewshot/mass_fs
[ -d $REPO/results/fewshot/isruc_fs ] && rsync -a $REPO/results/fewshot/isruc_fs/ $REPO/results/fewshot/isruc/ && rm -rf $REPO/results/fewshot/isruc_fs

python3 finalize/aggregate_local.py >> $LOG 2>&1 || true
git add -A
if git diff --cached --quiet; then
  echo "[$(date)] no changes" >> $LOG
else
  git commit -m "Auto-update results ($(date +'%F %H:%M'))" >> $LOG 2>&1
  git push origin main >> $LOG 2>&1 && echo "[$(date)] pushed" >> $LOG || echo "[$(date)] push failed" >> $LOG
fi
