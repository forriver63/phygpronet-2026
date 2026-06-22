#!/bin/bash
# Robust queue runner for PhyG ablation extension.
# Idempotent · auto-retries · auto-evals SHHS per-class · writes RESULTS.md
#
# Usage: nohup ./queue_runner.sh QUEUE.txt > /tmp/queue.out 2>&1 & disown
#
# QUEUE.txt format (one config per line, space-separated):
#   <name> <epochs> <mech_w> <warmup> <hcp> <sw> <sp> <ptk> <seed>
#
# Outputs:
#   $BASE/STATUS.txt          live counter (done/todo/fail/current)
#   $BASE/RESULTS.md          markdown table appended after each successful run
#   $BASE/queue_runner.log    detailed log

set -u
QUEUE="${1:-/root/autodl-tmp/queue.txt}"
PY=/root/miniconda3/bin/python
TRAIN_SCR=/root/autodl-tmp/train_phyg_v2.py
EVAL_SCR=/root/autodl-tmp/eval_shhs_perclass.py
DATA=/root/autodl-tmp/data/sleepedf_memmap_v2_mech
BASE=/root/autodl-tmp/runs/overnight
STATUS=$BASE/STATUS.txt
RESULTS_MD=$BASE/RESULTS.md
LOG=$BASE/queue_runner.log
HOSTNAME=$(hostname)

mkdir -p $BASE
echo "[start $(date)] host=$HOSTNAME queue=$QUEUE" >> $LOG

# Initialize markdown header if missing
if [ ! -f $RESULTS_MD ]; then
  cat > $RESULTS_MD <<'EOF'
# Ablation Extension Results (per-run)

每行 = 一次训练的 in-domain test + SHHS zero-shot 指标。
后续可用 aggregate_results.py 计算 mean ± std。

| config | seed | host | test_macroF1 | test_balAcc | test_kappa | test_W | test_N1 | test_N2 | test_N3 | test_REM | shhs_macroF1 | shhs_balAcc | shhs_kappa | shhs_W | shhs_N1 | shhs_N2 | shhs_N3 | shhs_REM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
EOF
fi

# Append one MD row from results.json + shhs_perclass.txt
append_md_row() {
  local name=$1 seed=$2 dir=$3
  $PY - <<PYEOF >> $RESULTS_MD
import json, os, re
d = json.load(open("$dir/results.json"))
t = d.get("test", d)
pf = t.get("per_f1", [0]*5)
row = ["$name", str($seed), "$HOSTNAME",
       f"{t['macro_f1']:.4f}", f"{t['bal_acc']:.4f}", f"{t['kappa']:.4f}",
       f"{pf[0]:.4f}", f"{pf[1]:.4f}", f"{pf[2]:.4f}", f"{pf[3]:.4f}", f"{pf[4]:.4f}"]
# SHHS from text file
shhs_path = "$dir/shhs_perclass.txt"
if os.path.exists(shhs_path):
    txt = open(shhs_path).read()
    m1 = re.search(r"macro_f1=([\d.]+)\s+bal_acc=([\d.]+)\s+kappa=([\d.]+)", txt)
    m2 = re.search(r"per_f1=\[W=([\d.]+),N1=([\d.]+),N2=([\d.]+),N3=([\d.]+),REM=([\d.]+)\]", txt)
    if m1: row += [m1.group(1), m1.group(2), m1.group(3)]
    else: row += ["", "", ""]
    if m2: row += [m2.group(i) for i in range(1, 6)]
    else: row += [""]*5
else:
    row += [""]*8
print("| " + " | ".join(row) + " |")
PYEOF
}

write_status() {
  local total=$(wc -l < $QUEUE)
  local done_count=0 todo_count=0 fail_count=0
  local _n _ep _mw _wu _hcp _sw _sp _ptk _sd _out
  while IFS=' ' read -r _n _ep _mw _wu _hcp _sw _sp _ptk _sd; do
    [ -z "$_n" ] && continue
    [[ $_n == \#* ]] && continue
    _out=$BASE/${_n}_seed${_sd}
    if [ -f $_out/results.json ]; then
      done_count=$((done_count+1))
    elif [ -f $_out/FAILED ]; then
      fail_count=$((fail_count+1))
    else
      todo_count=$((todo_count+1))
    fi
  done < $QUEUE
  cat > $STATUS <<EOF
[$(date)] host=$HOSTNAME
Queue: $QUEUE
  done: $done_count
  todo: $todo_count
  fail: $fail_count
  current: ${CURRENT:-idle}

RESULTS_MD: $RESULTS_MD
EOF
}

run_one() {
  local name=$1 ep=$2 mw=$3 wu=$4 hcp=$5 sww=$6 spw=$7 ptk=$8 seed=$9
  local out=$BASE/${name}_seed${seed}
  if [ -f $out/results.json ]; then
    echo "[skip] $name s$seed" >> $LOG
    return 0
  fi
  mkdir -p $out
  rm -f $out/FAILED
  CURRENT="$name s$seed"
  write_status

  for attempt in 1 2 3; do
    echo "[run] $name s$seed attempt=$attempt @ $(date)" >> $LOG
    $PY $TRAIN_SCR --packed_dir $DATA \
      --seed $seed --epochs $ep --K1 96 --tau 3.5 \
      --mech_w $mw --mech_warmup_epoch $wu --mech_high_conf_pct $hcp \
      --slowwave_w $sww --spindle_w $spw --pg_topk $ptk \
      --use_pg --num_workers 4 --out_dir $out >> $LOG 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f $out/results.json ]; then
      # Auto-eval SHHS per-class
      if [ ! -f $out/shhs_perclass.txt ] && [ -f $out/best_model.pt ] && [ -f $EVAL_SCR ]; then
        echo "[eval] $name s$seed shhs per-class @ $(date)" >> $LOG
        $PY $EVAL_SCR --ckpt $out/best_model.pt > $out/shhs_perclass.txt 2>&1 || true
      fi
      # Append markdown row
      append_md_row "$name" "$seed" "$out"
      echo "[done] $name s$seed @ $(date)" >> $LOG
      write_status
      return 0
    fi
    echo "[retry] $name s$seed attempt=$attempt exit=$rc @ $(date)" >> $LOG
    sleep 30
  done
  echo "[FAIL] $name s$seed after 3 attempts @ $(date)" >> $LOG
  touch $out/FAILED
  write_status
  return 1
}

while IFS=' ' read -r name ep mw wu hcp sww spw ptk seed; do
  [ -z "$name" ] && continue
  [[ $name == \#* ]] && continue
  run_one "$name" "$ep" "$mw" "$wu" "$hcp" "$sww" "$spw" "$ptk" "$seed"
done < $QUEUE

CURRENT="all done"
write_status
echo "[end $(date)] host=$HOSTNAME all done" >> $LOG
