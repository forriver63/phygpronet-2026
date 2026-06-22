"""Aggregate all RESULTS.md from B/C/D + already-existing winner data into a single summary.

Pulls per-row data from each machine's RESULTS.md and computes mean ± std grouped by config name.
Output: /Users/max/dev/icml/PhyG_Ablation_Summary.md (and console).
"""
import subprocess, re, statistics, json, os
from collections import defaultdict

HOSTS = ['autodl-b', 'autodl-c', 'autodl-d']
ROWS = []

def parse_md(host):
    r = subprocess.run(['ssh', host, 'cat /root/autodl-tmp/runs/overnight/RESULTS.md 2>/dev/null'],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        if not line.startswith('| ') or line.startswith('| ---') or line.startswith('| config'):
            continue
        parts = [p.strip() for p in line.strip('| ').split('|')]
        if len(parts) < 11: continue
        try:
            row = {
                'config': parts[0], 'seed': int(parts[1]), 'host': parts[2],
                'test_macro_f1': float(parts[3]), 'test_bal_acc': float(parts[4]), 'test_kappa': float(parts[5]),
                'test_W': float(parts[6]), 'test_N1': float(parts[7]), 'test_N2': float(parts[8]),
                'test_N3': float(parts[9]), 'test_REM': float(parts[10]),
            }
            if len(parts) >= 19:
                for k, v in zip(['shhs_macro_f1','shhs_bal_acc','shhs_kappa',
                                  'shhs_W','shhs_N1','shhs_N2','shhs_N3','shhs_REM'], parts[11:19]):
                    row[k] = float(v) if v else None
            out.append(row)
        except Exception as e:
            pass
    return out

for h in HOSTS:
    ROWS += parse_md(h)

# Also pull existing PhyG winner (seed 42-46) from local final_results.json
fr_path = '/Users/max/dev/icml/phygpronet-2026/results/final_results.json'
if os.path.exists(fr_path):
    R = json.load(open(fr_path))
    for s, d in R.get('phyg_winner', {}).get('seeds', {}).items():
        seed = int(s.replace('seed', ''))
        td = d.get('in_domain', {}); pf = td.get('per_f1', [0]*5)
        sd = d.get('shhs_zs', {}); spf = sd.get('per_f1', [0]*5) if sd else [0]*5
        ROWS.append({
            'config': 'exp3_hc20', 'seed': seed, 'host': 'historical',
            'test_macro_f1': td['macro_f1'], 'test_bal_acc': td['bal_acc'], 'test_kappa': td['kappa'],
            'test_W': pf[0], 'test_N1': pf[1], 'test_N2': pf[2], 'test_N3': pf[3], 'test_REM': pf[4],
            'shhs_macro_f1': sd.get('macro_f1') if sd else None,
            'shhs_bal_acc': sd.get('bal_acc') if sd else None,
            'shhs_kappa': sd.get('kappa') if sd else None,
            'shhs_W': spf[0], 'shhs_N1': spf[1], 'shhs_N2': spf[2], 'shhs_N3': spf[3], 'shhs_REM': spf[4],
        })

print(f"Loaded {len(ROWS)} rows from {len(HOSTS)} hosts + historical")

# Group by config
groups = defaultdict(list)
for r in ROWS:
    groups[r['config']].append(r)

def ms(arr):
    arr = [x for x in arr if x is not None]
    if not arr: return ('—', '—')
    m = statistics.mean(arr)
    s = statistics.stdev(arr) if len(arr) > 1 else 0
    return f'{m:.4f}', f'{s:.4f}'

# Build output
lines = []
lines.append('# PhyG Ablation 全部 seeds 聚合 (B + C + D + winner historical)\n')
lines.append(f'更新时间：每次跑 aggregate_results.py 自动重新计算\n')
lines.append('每个 cell = mean ± std，n = 跨 seeds 的有效次数。\n\n')

# Master table: in-domain
lines.append('## A. Sleep-EDF in-domain test\n')
lines.append('| config | n | macroF1 | balAcc | kappa | W | N1 | N2 | N3 | REM |')
lines.append('|---|---|---|---|---|---|---|---|---|---|')
for cfg in sorted(groups.keys()):
    rs = groups[cfg]
    n = len(rs)
    m_mf1, s_mf1 = ms([r['test_macro_f1'] for r in rs])
    m_ba, s_ba   = ms([r['test_bal_acc'] for r in rs])
    m_k, s_k     = ms([r['test_kappa'] for r in rs])
    cells = [cfg, str(n), f'{m_mf1}±{s_mf1}', f'{m_ba}±{s_ba}', f'{m_k}±{s_k}']
    for col in ['test_W','test_N1','test_N2','test_N3','test_REM']:
        mm, ss = ms([r[col] for r in rs])
        cells.append(f'{mm}±{ss}')
    lines.append('| ' + ' | '.join(cells) + ' |')

# SHHS table
lines.append('\n## B. SHHS zero-shot\n')
lines.append('| config | n | macroF1 | balAcc | kappa | W | N1 | N2 | N3 | REM |')
lines.append('|---|---|---|---|---|---|---|---|---|---|')
for cfg in sorted(groups.keys()):
    rs = groups[cfg]
    n_shhs = sum(1 for r in rs if r.get('shhs_macro_f1') is not None)
    m_mf1, s_mf1 = ms([r.get('shhs_macro_f1') for r in rs])
    m_ba, s_ba   = ms([r.get('shhs_bal_acc') for r in rs])
    m_k, s_k     = ms([r.get('shhs_kappa') for r in rs])
    cells = [cfg, str(n_shhs), f'{m_mf1}±{s_mf1}', f'{m_ba}±{s_ba}', f'{m_k}±{s_k}']
    for col in ['shhs_W','shhs_N1','shhs_N2','shhs_N3','shhs_REM']:
        mm, ss = ms([r.get(col) for r in rs])
        cells.append(f'{mm}±{ss}')
    lines.append('| ' + ' | '.join(cells) + ' |')

# Per-config per-seed detail
lines.append('\n## C. Per-seed 明细\n')
for cfg in sorted(groups.keys()):
    rs = sorted(groups[cfg], key=lambda r: r['seed'])
    lines.append(f'\n### {cfg} (n={len(rs)})\n')
    lines.append('| seed | host | test_macroF1 | test_N3 | shhs_macroF1 | shhs_N3 |')
    lines.append('|---|---|---|---|---|---|')
    for r in rs:
        sm = f"{r['shhs_macro_f1']:.4f}" if r.get('shhs_macro_f1') else '—'
        sn3 = f"{r.get('shhs_N3'):.4f}" if r.get('shhs_N3') is not None else '—'
        lines.append(f"| {r['seed']} | {r['host']} | {r['test_macro_f1']:.4f} | {r['test_N3']:.4f} | {sm} | {sn3} |")

OUT = '/Users/max/dev/icml/PhyG_Ablation_Summary.md'
open(OUT, 'w').write('\n'.join(lines))
print(f'Wrote {OUT} ({sum(len(g) for g in groups.values())} runs across {len(groups)} configs)')
