#!/bin/bash
# Run-anywhere finalize: aggregate results from 3 AutoDL cards onto ANY machine.
#
# Usage on a fresh machine (no Mac, no Claude):
#   git clone https://github.com/forriver63/phygpronet-2026.git
#   cd phygpronet-2026
#   bash finalize/run_anywhere.sh
#
# Prerequisites:
#   - SSH key access to the 3 AutoDL cards
#   - python3 installed
#   - (optional) gh CLI for git push
#
# If your SSH config doesn't have autodl-b/c/d aliases yet, set these env vars:
#   export AUTODL_B="root@connect.westb.seetacloud.com -p <port>"
#   export AUTODL_C="root@connect.westc.seetacloud.com -p <port>"
#   export AUTODL_D="root@connect.westb.seetacloud.com -p <port>"
# Defaults try ssh aliases: autodl-b, autodl-c, autodl-d.

set -e
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
echo "REPO=$REPO_ROOT"
cd "$REPO_ROOT"

# Resolve host args
B="${AUTODL_B:-autodl-b}"
C="${AUTODL_C:-autodl-c}"
D="${AUTODL_D:-autodl-d}"
HOST_KEYS=(B C D)
HOST_VALS=("$B" "$C" "$D")

echo "=== Step 1: Pull RESULTS.md + STATUS.txt from each card ==="
mkdir -p results/ablations/per_card
for i in 0 1 2; do
  k=${HOST_KEYS[$i]}; h=${HOST_VALS[$i]}
  echo "  $k ($h) ..."
  ssh $h "cat /root/autodl-tmp/runs/overnight/RESULTS.md 2>/dev/null" \
    > results/ablations/per_card/${k}_RESULTS.md 2>/dev/null || echo "    (skip)"
  ssh $h "cat /root/autodl-tmp/runs/overnight/STATUS.txt 2>/dev/null" \
    > results/ablations/per_card/${k}_STATUS.txt 2>/dev/null || true
done

echo
echo "=== Step 2: Pull per-config results.json (idempotent tar) ==="
for i in 0 1 2; do
  k=${HOST_KEYS[$i]}; h=${HOST_VALS[$i]}
  echo "  $k ..."
  ssh $h 'cd /root/autodl-tmp/runs/overnight && find . -maxdepth 2 -name results.json | head -300 | xargs -r tar cf - 2>/dev/null' \
    | tar xf - -C results/ablations/ 2>/dev/null || echo "    (skip)"
done

echo
echo "=== Step 3: Aggregate mean ± std ==="
python3 finalize/aggregate_local.py

echo
echo "=== Step 4: Git commit + push (optional) ==="
git add -A
if git diff --cached --quiet; then
  echo "  nothing to commit"
else
  GIT_USER="${GIT_USER_NAME:-forriver63}"
  GIT_EMAIL="${GIT_USER_EMAIL:-t47302088@gmail.com}"
  git -c user.email="$GIT_EMAIL" -c user.name="$GIT_USER" \
    commit -m "Auto-update results ($(date +%Y-%m-%d))"
  if command -v gh >/dev/null 2>&1; then
    gh auth switch -u "$GIT_USER" 2>/dev/null || true
  fi
  git push origin main || echo "  (push failed — run 'git push origin main' manually after fixing auth)"
fi

echo
echo "=== Done ==="
echo "Summary: $REPO_ROOT/results/ablations/SUMMARY.md"
echo "GitHub:  https://github.com/forriver63/phygpronet-2026"
