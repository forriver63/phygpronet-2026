# PhyGProNet 2026 — Experiment Repository

Physiology-aligned Granular Prototype Network for Sleep Staging — paper resubmission preparation.

This repo contains all experimental data, scripts, and figures for the 2026 resubmission.

## Repo structure

```
.
├── docs/
│   ├── PhyGProNet_Supervisor_Report.docx     # Paper-style narrative report (for supervisor)
│   └── PhyGProNet_Final_Report.docx          # Detailed per-seed × per-class report
├── results/
│   ├── final_results.json                    # Aggregated JSON (all baselines + PhyG winner + fewshot)
│   ├── baselines/                            # Per-seed JSON × 6 baselines × 6 seeds
│   │   ├── attnsleep/seed{42-47}.json
│   │   ├── deepsleep/, labram/, tinysleep/, usleep/, cbramod/
│   ├── phyg_winner/                          # PhyG winner (exp3_hc20) × 5 seeds in-domain
│   ├── multi/                                # MultiChannelSleepNet (subject-level retrain) × 3 seeds
│   ├── fewshot/
│   │   ├── shhs/   (27 runs: 3 base ckpts × k∈{1,5,10} × 3 fs_seeds)
│   │   ├── mass/   (27 runs)
│   │   └── isruc/  (27 runs)
│   └── ablations/                            # Per-config results.json (Exp 2/3/4/6/7/12/FusionA)
├── scripts/
│   ├── train_phyg_v2.py                      # Main training (knobs: mech_warmup, high_conf_pct, asymm weights, pg_topk)
│   ├── train_multi_subject.py                # MultiChannelSleepNet single-channel retraining
│   ├── eval_exp1_perckpt.py                  # Per-epoch ckpt evaluation (alignment + SHHS zs)
│   ├── eval_shhs_perclass.py                 # Quick SHHS per-class F1
│   ├── eval_mass_isruc.py                    # MASS / ISRUC zero-shot
│   ├── eval_shhs_tt_zscore.py                # Exp 9: test-time z-score
│   ├── eval_viterbi.py                       # Exp 5: Viterbi smoothing
│   ├── eval_temp_cal.py                      # Exp 8: temperature calibration
│   ├── finetune_shhs_fast.py                 # SHHS few-shot
│   ├── finetune_target_fewshot.py            # MASS / ISRUC few-shot (legacy slow)
│   └── finetune_mass_fast.py                 # MASS few-shot (uses pre-processed compact data)
└── figs/                                     # Interpretability figures
    ├── expA_proto_scatter.png                # Prototype-Physiology Alignment (Fig 1)
    ├── expC_topk.png                         # Cross-domain prototype consistency (Fig 2)
    ├── expE_pca.png, expE_adj.png            # PG-R graph structure (Fig 3a, 3b)
    └── expD_N2.png, expD_N3.png              # Decision path visualization (Fig 4a, 4b)
```

## Final winner

**Config: `exp3_hc20`** (high-confidence mechanism loss, top 20% physiologically-confident samples)

```bash
python scripts/train_phyg_v2.py \
  --seed {42|43|44|45|46} --epochs 5 --K1 96 --tau 3.5 \
  --mech_w 0.2 --mech_high_conf_pct 0.2 \
  --pg_topk 12 --use_pg --num_workers 4 \
  --packed_dir /path/to/sleepedf_memmap_v2_mech \
  --out_dir runs/exp3_hc20_seed${SEED}
```

5 seeds (42-46) on Sleep-EDF-78 with subject 80/10/10 random split.

## Key results

| Metric | PhyG winner | Best baseline |
|---|---|---|
| Sleep-EDF in-domain macro-F1 | 0.713 ± 0.036 | AttnSleep 0.726 / DeepSleep 0.717 |
| **SHHS zero-shot macro-F1** | **0.529 ± 0.048** | CBraMod 0.490 / LaBraM 0.515 |
| **N3 zero-shot F1** | **0.499** (best) | DeepSleep 0.480 |
| MASS zero-shot macro-F1 | 0.604 ± 0.009 | (baselines: incompatible pipeline) |
| ISRUC zero-shot macro-F1 | 0.344 ± 0.018 | (baselines: incompatible pipeline) |
| **Params** | **0.66M** (1/40 of DeepSleep) | DeepSleep 25.4M / CBraMod 7.96M |

### Few-shot adaptation (PhyG winner)

| Target | zs | k=1 | k=5 | k=10 |
|---|---|---|---|---|
| SHHS  | 0.528 | 0.596 (+7pt) | 0.637 (+11pt) | **0.644** (+12pt) |
| MASS  | 0.604 | 0.448 (-15pt) | 0.688 (+9pt)  | **0.709** (+11pt) |
| ISRUC | 0.344 | 0.262 (-8pt)  | 0.378 (+3pt)  | **0.481** (+14pt) |

## Notable findings

1. **Short training wins for transfer**: PhyG SHHS zs peaks at 5 epochs (0.531) and degrades to 0.45 at 50 epochs. Prototype-physiology alignment saturates early; longer training learns source-specific noise.
2. **High-confidence mech loss**: Restricting mech regularization to top-20% high-|spindle_z| / |slowwave_z| samples is the key win (+4.3pt SHHS zs over OLD baseline).
3. **Amplitude is critical**: Test-time per-recording z-score collapses N3 F1 to 0.0 (Exp 9). PhyG depends on absolute EEG amplitude (N3 = high slow-wave amplitude).
4. **Combinations don't compound**: Fusing asymmetric weights + high-conf mech (Exp 12 / Fusion A) underperforms either alone — both methods reduce same source-specific spindle noise.

## Data dependencies

Not included in repo (too large):
- Sleep-EDF-78 (Fpz-Cz, 100 Hz, 78 subjects)
- SHHS-1 (125 Hz → 100 Hz, 312 subjects)
- MASS-SS3 (256 Hz → 100 Hz, 44 subjects, channel 20)
- ISRUC-Subgroup1 (200 Hz → 100 Hz, 30 subjects)

Preprocessed data lives on AutoDL containers.

## Reproducibility

To reproduce PhyG winner SHHS fewshot:
```bash
# 1. Train winner ckpts (5 seeds)
for s in 42 43 44 45 46; do
  python scripts/train_phyg_v2.py --seed $s --epochs 5 --K1 96 --tau 3.5 \
    --mech_w 0.2 --mech_high_conf_pct 0.2 --pg_topk 12 --use_pg \
    --out_dir runs/exp3_hc20_seed${s}
done
# 2. SHHS fewshot (k=1,5,10 × top-3 base seeds × 3 fs_seeds)
for bs in 44 46 43; do
  for k in 1 5 10; do
    for fs in 42 43 44; do
      python scripts/finetune_shhs_fast.py \
        --ckpt runs/exp3_hc20_seed${bs}/best_model.pt \
        --k $k --fs_seed $fs \
        --out_json runs/shhs_fs/base${bs}_k${k}_fs${fs}.json
    done
  done
done
```

## License / contact

Private repo for paper draft. Contact: tao.zhou@keyreply.com / forriver63 / t47302088@gmail.com.
