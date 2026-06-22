"""Aggregate per-card RESULTS.md into a single mean ± std summary.
Reads from <repo>/results/ablations/per_card/*_RESULTS.md and historical final_results.json.
Writes <repo>/results/ablations/SUMMARY.md.
"""
import os, json, statistics, glob
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(REPO, 'results')
OUT_PATH = os.path.join(RESULTS_DIR, 'ablations', 'SUMMARY.md')

ROWS = []

for f in sorted(glob.glob(os.path.join(RESULTS_DIR, 'ablations', 'per_card', '*_RESULTS.md'))):
    host = os.path.basename(f).replace('_RESULTS.md', '')
    for line in open(f):
        if not line.startswith('| ') or line.startswith('| ---') or line.startswith('| config'):
            continue
        parts = [p.strip() for p in line.strip('| \n').split('|')]
        if len(parts) < 11: continue
        try:
            row = {'config': parts[0], 'seed': int(parts[1]), 'host': host,
                   'test_macro_f1': float(parts[3]), 'test_bal_acc': float(parts[4]),
                   'test_kappa': float(parts[5]),
                   'test_W': float(parts[6]), 'test_N1': float(parts[7]),
                   'test_N2': float(parts[8]), 'test_N3': float(parts[9]),
                   'test_REM': float(parts[10])}
            if len(parts) >= 19:
                for k, v in zip(['shhs_macro_f1','shhs_bal_acc','shhs_kappa',
                                  'shhs_W','shhs_N1','shhs_N2','shhs_N3','shhs_REM'], parts[11:19]):
                    row[k] = float(v) if v else None
            ROWS.append(row)
        except Exception:
            pass

fr = os.path.join(RESULTS_DIR, 'final_results.json')
if os.path.exists(fr):
    R = json.load(open(fr))
    for s, d in R.get('phyg_winner', {}).get('seeds', {}).items():
        seed = int(s.replace('seed', ''))
        td = d.get('in_domain', {}); pf = td.get('per_f1', [0]*5)
        sd = d.get('shhs_zs', {}) or {}; spf = sd.get('per_f1', [0]*5)
        ROWS.append({
            'config': 'exp3_hc20', 'seed': seed, 'host': 'historical',
            'test_macro_f1': td['macro_f1'], 'test_bal_acc': td['bal_acc'], 'test_kappa': td['kappa'],
            'test_W': pf[0], 'test_N1': pf[1], 'test_N2': pf[2], 'test_N3': pf[3], 'test_REM': pf[4],
            'shhs_macro_f1': sd.get('macro_f1'), 'shhs_bal_acc': sd.get('bal_acc'),
            'shhs_kappa': sd.get('kappa'),
            'shhs_W': spf[0], 'shhs_N1': spf[1], 'shhs_N2': spf[2], 'shhs_N3': spf[3], 'shhs_REM': spf[4],
        })

print(f"parsed {len(ROWS)} runs")

groups = defaultdict(list)
for r in ROWS:
    groups[r['config']].append(r)

def ms(arr):
    arr = [x for x in arr if x is not None]
    if not arr: return ('—', '—')
    m = statistics.mean(arr); s = statistics.stdev(arr) if len(arr) > 1 else 0
    return f'{m:.4f}', f'{s:.4f}'

lines = ['# PhyG Ablation 全部 seeds 聚合\n',
         '更新时间：每次 bash finalize/run_anywhere.sh 自动重新计算\n\n',
         '## A. Sleep-EDF in-domain test\n',
         '| config | n | macroF1 | balAcc | kappa | W | N1 | N2 | N3 | REM |',
         '|---|---|---|---|---|---|---|---|---|---|']
for cfg in sorted(groups):
    rs = groups[cfg]; n = len(rs)
    m1,s1 = ms([r['test_macro_f1'] for r in rs])
    m2,s2 = ms([r['test_bal_acc'] for r in rs])
    m3,s3 = ms([r['test_kappa'] for r in rs])
    cells = [cfg, str(n), f'{m1}±{s1}', f'{m2}±{s2}', f'{m3}±{s3}']
    for col in ['test_W','test_N1','test_N2','test_N3','test_REM']:
        mm,ss = ms([r[col] for r in rs])
        cells.append(f'{mm}±{ss}')
    lines.append('| ' + ' | '.join(cells) + ' |')

lines += ['\n## B. SHHS zero-shot\n',
          '| config | n | macroF1 | balAcc | kappa | W | N1 | N2 | N3 | REM |',
          '|---|---|---|---|---|---|---|---|---|---|']
for cfg in sorted(groups):
    rs = groups[cfg]
    n_shhs = sum(1 for r in rs if r.get('shhs_macro_f1') is not None)
    m1,s1 = ms([r.get('shhs_macro_f1') for r in rs])
    m2,s2 = ms([r.get('shhs_bal_acc') for r in rs])
    m3,s3 = ms([r.get('shhs_kappa') for r in rs])
    cells = [cfg, str(n_shhs), f'{m1}±{s1}', f'{m2}±{s2}', f'{m3}±{s3}']
    for col in ['shhs_W','shhs_N1','shhs_N2','shhs_N3','shhs_REM']:
        mm,ss = ms([r.get(col) for r in rs])
        cells.append(f'{mm}±{ss}')
    lines.append('| ' + ' | '.join(cells) + ' |')

lines.append('\n## C. Per-seed 明细\n')
for cfg in sorted(groups):
    rs = sorted(groups[cfg], key=lambda r: r['seed'])
    lines.append(f'\n### {cfg} (n={len(rs)})\n')
    lines.append('| seed | host | test_macroF1 | test_N3 | shhs_macroF1 | shhs_N3 |')
    lines.append('|---|---|---|---|---|---|')
    for r in rs:
        sm = f"{r['shhs_macro_f1']:.4f}" if r.get('shhs_macro_f1') else '—'
        sn3 = f"{r.get('shhs_N3'):.4f}" if r.get('shhs_N3') is not None else '—'
        lines.append(f"| {r['seed']} | {r['host']} | {r['test_macro_f1']:.4f} | {r['test_N3']:.4f} | {sm} | {sn3} |")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
open(OUT_PATH, 'w').write('\n'.join(lines))
print(f"wrote {OUT_PATH}")
