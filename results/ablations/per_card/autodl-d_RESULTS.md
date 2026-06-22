# Ablation Extension Results (per-run)

每行 = 一次训练的 in-domain test + SHHS zero-shot 指标。
后续可用 aggregate_results.py 计算 mean ± std。

| config | seed | host | test_macroF1 | test_balAcc | test_kappa | test_W | test_N1 | test_N2 | test_N3 | test_REM | shhs_macroF1 | shhs_balAcc | shhs_kappa | shhs_W | shhs_N1 | shhs_N2 | shhs_N3 | shhs_REM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp2_wu3_ep10 | 47 | autodl-container-320f4f8aac-3cbf9748 | 0.7476 | 0.7224 | 0.8121 | 0.9761 | 0.3212 | 0.8543 | 0.8749 | 0.7114 | 0.4980 | 0.5080 | 0.4887 | 0.736 | 0.041 | 0.697 | 0.372 | 0.644 |
| exp2_wu5_ep15 | 47 | autodl-container-320f4f8aac-3cbf9748 | 0.7560 | 0.7445 | 0.8231 | 0.9814 | 0.3275 | 0.8396 | 0.8891 | 0.7423 | 0.4969 | 0.5064 | 0.4681 | 0.762 | 0.049 | 0.655 | 0.380 | 0.638 |
