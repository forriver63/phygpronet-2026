#!/bin/bash
# Wait for queue_runner.sh (ablation) to finish, then start fewshot_runner.
QUEUE_FS=${1:?need fewshot queue file}
LOG=/root/autodl-tmp/runs/overnight/chain_fewshot.log
echo "[$(date)] waiting for ablation queue_runner to finish ..." >> $LOG
while pgrep -f 'queue_runner.sh' >/dev/null 2>&1; do sleep 60; done
echo "[$(date)] ablation done, launching fewshot $QUEUE_FS ..." >> $LOG
nohup /root/autodl-tmp/fewshot_runner.sh $QUEUE_FS > /tmp/fs.out 2>&1
echo "[$(date)] fewshot done" >> $LOG
