# Ablation Extension Results (per-run)

每行 = 一次训练的 in-domain test + SHHS zero-shot 指标。
后续可用 aggregate_results.py 计算 mean ± std。

| config | seed | host | test_macroF1 | test_balAcc | test_kappa | test_W | test_N1 | test_N2 | test_N3 | test_REM | shhs_macroF1 | shhs_balAcc | shhs_kappa | shhs_W | shhs_N1 | shhs_N2 | shhs_N3 | shhs_REM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp2_wu3_ep10 | 45 | autodl-container-b0bf4cb35f-44866196 | 0.7362 | 0.7487 | 0.7929 | 0.9736 | 0.4147 | 0.8200 | 0.7819 | 0.6906 | 0.5143 | 0.5128 | 0.4965 | 0.711 | 0.069 | 0.711 | 0.427 | 0.654 |
| exp2_wu5_ep15 | 45 | autodl-container-b0bf4cb35f-44866196 | 0.7077 | 0.7107 | 0.7852 | 0.9748 | 0.2779 | 0.8077 | 0.7712 | 0.7067 | 0.5291 | 0.5154 | 0.5275 | 0.693 | 0.040 | 0.754 | 0.481 | 0.677 |
| exp3_hc30 | 45 | autodl-container-b0bf4cb35f-44866196 | 0.6913 | 0.6978 | 0.7724 | 0.9737 | 0.2811 | 0.7922 | 0.7879 | 0.6217 | 0.5280 | 0.5094 | 0.5222 | 0.658 | 0.031 | 0.758 | 0.530 | 0.662 |
