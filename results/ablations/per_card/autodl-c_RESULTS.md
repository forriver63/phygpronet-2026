# Ablation Extension Results (per-run)

每行 = 一次训练的 in-domain test + SHHS zero-shot 指标。
后续可用 aggregate_results.py 计算 mean ± std。

| config | seed | host | test_macroF1 | test_balAcc | test_kappa | test_W | test_N1 | test_N2 | test_N3 | test_REM | shhs_macroF1 | shhs_balAcc | shhs_kappa | shhs_W | shhs_N1 | shhs_N2 | shhs_N3 | shhs_REM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp2_wu3_ep10 | 46 | autodl-container-63d44c988c-902f451b | 0.7602 | 0.7492 | 0.8415 | 0.9870 | 0.4243 | 0.8806 | 0.7912 | 0.7177 | 0.4976 | 0.5037 | 0.4829 | 0.738 | 0.073 | 0.708 | 0.296 | 0.674 |
| exp2_wu5_ep15 | 46 | autodl-container-63d44c988c-902f451b | 0.7607 | 0.7600 | 0.8486 | 0.9865 | 0.3646 | 0.8862 | 0.7963 | 0.7700 | 0.5122 | 0.5099 | 0.4801 | 0.704 | 0.088 | 0.714 | 0.403 | 0.652 |
| exp3_hc30 | 46 | autodl-container-63d44c988c-902f451b | 0.7602 | 0.7545 | 0.8388 | 0.9869 | 0.4328 | 0.8808 | 0.7843 | 0.7163 | 0.5043 | 0.5084 | 0.4957 | 0.714 | 0.036 | 0.711 | 0.382 | 0.678 |
