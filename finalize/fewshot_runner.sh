#!/bin/bash
# Few-shot queue runner — extends fewshot from 3 to 6 base seeds.
#
# Usage: nohup ./fewshot_runner.sh QUEUE.txt > /tmp/fs.out 2>&1 & disown
#
# Reads QUEUE.txt with lines: <target> <base_seed> <k> <fs_seed>
# Writes per-run JSON at /root/autodl-tmp/runs/overnight/{shhs_fs2,mass_fs,isruc_fs}/baseN_kK_fsM.json
# Idempotent — skips configs whose output JSON already exists.

set -u
QUEUE="${1:-/root/autodl-tmp/queue_fewshot.txt}"
PY=/root/miniconda3/bin/python
SHHS_SCR=/root/autodl-tmp/finetune_shhs_fast.py
MASS_SCR=/root/autodl-tmp/finetune_mass_fast.py
ISRUC_SCR=/root/autodl-tmp/finetune_target_fewshot.py
BASE=/root/autodl-tmp/runs/overnight
LOG=$BASE/fewshot_runner.log
HOSTNAME=$(hostname)

mkdir -p $BASE/shhs_fs2 $BASE/mass_fs $BASE/isruc_fs
echo "[fewshot start $(date)] host=$HOSTNAME queue=$QUEUE" >> $LOG

run_one() {
  local tgt=$1 bs=$2 k=$3 fs=$4
  local ckpt=$BASE/exp3_hc20_seed${bs}/best_model.pt
  if [ ! -f $ckpt ]; then echo "[skip] no ckpt for s$bs"; return 1; fi
  local out
  case $tgt in
    shhs)  out=$BASE/shhs_fs2/base${bs}_k${k}_fs${fs}.json ;;
    mass)  out=$BASE/mass_fs/base${bs}_k${k}_fs${fs}.json ;;
    isruc) out=$BASE/isruc_fs/base${bs}_k${k}_fs${fs}.json ;;
    *) echo "[err] unknown target $tgt"; return 1 ;;
  esac
  if [ -f $out ]; then echo "[skip] $tgt base$bs k$k fs$fs" >> $LOG; return 0; fi
  echo "[run] $tgt base$bs k$k fs$fs @ $(date)" >> $LOG
  case $tgt in
    shhs)  $PY $SHHS_SCR --ckpt $ckpt --k $k --fs_seed $fs --out_json $out >> $LOG 2>&1 ;;
    mass)
      if [ -d /root/autodl-tmp/data/mass_npz_small ]; then
        $PY $MASS_SCR --ckpt $ckpt --k $k --fs_seed $fs --out_json $out >> $LOG 2>&1
      else
        $PY $ISRUC_SCR --ckpt $ckpt --target mass --target_dir /root/autodl-tmp/data/mass_npz --k $k --fs_seed $fs --out_json $out >> $LOG 2>&1
      fi
      ;;
    isruc) $PY $ISRUC_SCR --ckpt $ckpt --target isruc --target_dir /root/autodl-tmp/data/isruc_npz --k $k --fs_seed $fs --out_json $out >> $LOG 2>&1 ;;
  esac
  echo "[done] $tgt base$bs k$k fs$fs exit=$? @ $(date)" >> $LOG
}

while IFS=' ' read -r tgt bs k fs; do
  [ -z "$tgt" ] && continue
  [[ $tgt == \#* ]] && continue
  run_one "$tgt" "$bs" "$k" "$fs"
done < $QUEUE

echo "[fewshot end $(date)]" >> $LOG
