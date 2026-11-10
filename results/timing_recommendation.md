# Timing and memory test (2026-10-28 08:08)

Machine: cores=2, RAM total 12.7 GB, RAM budget 9.0 GB. Synthetic data, sizes 1,000,000, 4,000,000 rows x 10 columns. Extrapolated to 14,000,000 training rows (time: log-log fit, RAM: linear fit; rough).

| model | est. fit time (1 core) | est. peak RAM |
|---|---|---|
| LR | 2.1 min (1 core) | 3.6 GB |
| DT | 3.3 min (1 core) | 1.2 GB |
| RF | 76.0 min (100 full trees, 1 core) | 2.1 GB |

## Random Forest candidates (budget 60 min per fit, 9 GB RAM)

| n_estimators | max_samples | est. wall time (min) | est. RAM (GB) | fits budget |
|---|---|---|---|---|
| 100 | null (full) | 48 | 2.8 | yes |
| 100 | 0.5 | 21 | 2.5 | yes |
| 100 | 0.25 | 9 | 2.3 | yes |
| 50 | 0.25 | 5 | 2.2 | yes |
| 100 | 0.1 | 3 | 2.2 | yes |
| 50 | 0.1 | 2 | 2.1 | yes |
| 25 | 0.1 | 1 | 2.1 | yes |

**Recommendation:** n_estimators=100, max_samples=null (first candidate inside both budgets). Put these in config.yaml, then set frozen: true. Use the same values for P1 and P2.

Caveat: synthetic data only has the right shape. Re-check with the real number of signatures from the audit (H3).
