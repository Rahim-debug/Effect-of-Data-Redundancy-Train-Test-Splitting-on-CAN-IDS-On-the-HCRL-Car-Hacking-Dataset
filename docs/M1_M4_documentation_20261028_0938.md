# FYDP Sprint 1: models track, tasks M1 to M4 - complete documentation

Generated: 2026-10-28 09:37 (machine clock) by `src/make_documentation.py`. Sections marked **LIVE** were read or executed at that moment.

## 1. Where everything is saved

Project folder (absolute path on this machine):

`/content/drive/MyDrive/FYDP/files/fydp_models_sameer/fydp_models`

On Google Drive this is the same folder: open drive.google.com and follow `FYDP` > `files` > `fydp_models_sameer` > `fydp_models` (the exact path above, after `/MyDrive/`).

**LIVE file inventory** (what exists now, size, last modified):

| File (relative to project folder) | Size | Last modified |
|---|---|---|
| .pytest_cache/.gitignore | 0.0 KB | 2026-10-27 08:45 |
| .pytest_cache/CACHEDIR.TAG | 0.2 KB | 2026-10-27 08:45 |
| .pytest_cache/README.md | 0.3 KB | 2026-10-27 08:45 |
| .pytest_cache/v/cache/nodeids | 1.1 KB | 2026-10-28 09:13 |
| README.md | 0.7 KB | 2026-10-27 08:18 |
| config.yaml | 1.4 KB | 2026-10-28 08:14 |
| conftest.py | 0.0 KB | 2026-10-27 08:18 |
| docs/SAMEER_GUIDE.md | 7.9 KB | 2026-10-27 08:18 |
| requirements.txt | 0.1 KB | 2026-10-27 08:18 |
| results/timing_extrapolation.csv | 0.1 KB | 2026-10-28 08:08 |
| results/timing_recommendation.md | 1.1 KB | 2026-10-28 08:08 |
| results/timing_table.csv | 0.6 KB | 2026-10-28 08:08 |
| src/__init__.py | 0.0 KB | 2026-10-27 08:18 |
| src/common.py | 2.4 KB | 2026-10-27 08:18 |
| src/make_documentation.py | 20.6 KB | 2026-10-28 09:36 |
| src/metrics.py | 3.8 KB | 2026-10-27 08:18 |
| src/models.py | 1.4 KB | 2026-10-27 08:18 |
| src/run_experiments.py | 9.6 KB | 2026-10-27 08:18 |
| src/synthetic.py | 3.8 KB | 2026-10-27 08:18 |
| src/timing_test.py | 10.6 KB | 2026-10-27 08:18 |
| tests/test_metrics.py | 3.2 KB | 2026-10-27 08:18 |
| tests/test_models.py | 1.7 KB | 2026-10-27 08:18 |
| tests/test_runner.py | 2.4 KB | 2026-10-27 08:18 |

What each part is for:

| Location | Purpose | Task |
|---|---|---|
| src/common.py | Fixed class order, 10 feature names, seeds, protocol names, config hash, data loader | shared |
| src/metrics.py | Evaluation metrics written from the formulas | M1 |
| tests/test_metrics.py | 8 tests: metrics must equal scikit-learn | M1 |
| src/models.py | Builders for LR, DT, RF | M2 |
| config.yaml | All hyperparameters and the two safety flags (frozen, tolerance_fixed) | M2 / M3 |
| tests/test_models.py | 4 tests: seeds, class weights, scaler fitted on training side only | M2 |
| src/timing_test.py | Timing and RAM test on synthetic data, extrapolation, RF recommendation | M3 |
| results/timing_table.csv, timing_extrapolation.csv, timing_recommendation.md | Output of the real timing test | M3 |
| src/run_experiments.py | Experiment runner: models x protocols x seeds -> results/runs.csv | M4 |
| tests/test_runner.py | 5 tests: columns, resume, invalid splits, real-run guard, label mapping | M4 |
| src/synthetic.py | Synthetic CAN-like data and DUMMY splits, only for tests and dry runs | M3 / M4 |
| results/runs.csv (future) | Real results, appended by the runner (does not exist yet) | M5 |
| docs/ | This documentation and SAMEER_GUIDE.md | all |

## 2. Environment (LIVE)

| Item | Value |
|---|---|
| Python | 3.13.18 |
| Platform | Linux-6.6.159+-x86_64-with-glibc2.39 |
| CPU cores | 2 |
| RAM total | 12.7 GB |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| sklearn | 1.6.1 |
| yaml | 6.0.3 |
| pyarrow | 23.0.1 |

## 3. What happened, in order

| Date | What | Result |
|---|---|---|
| 22 to 26 Oct 2026 | Sprint 1 started late; work split tracker and proposal read | Plan: M1 to M5 for the models track |
| 27 Oct 2026 | M1 to M4 code written and tested (17 tests); tools set up in Google Colab with the project on Drive | All tests passed |
| 27 Oct 2026 | Small timing check (20k and 60k rows) on Colab to prove the script runs | Worked; numbers not used |
| 28 Oct 2026 | Real timing test on 1,000,000 and 4,000,000 synthetic rows, target 14,000,000 | See section 7 |
| 28 Oct 2026 | RF settings chosen and written into config.yaml, frozen set to true | See section 6 |
| 28 Oct 2026 | Tests re-run and runner dry run executed twice (second time all runs skipped = resume works) | 17 passed; resume confirmed |
| Pending | M5 first real run | Blocked: frames.parquet, split files, supervisor decision S3 (tolerance) |

## 4. Task by task

### M1 - Metrics (`src/metrics.py`, `tests/test_metrics.py`)

**What:** the evaluation numbers of the proposal, written from the formulas and not imported from a library.

**Formulas used** (rows of the confusion matrix = true class, columns = predicted class):

- precision_c = TP / (TP + FP); recall_c = TP / (TP + FN); F1_c = 2TP / (2TP + FP + FN)
- macro-F1 = (1/C) x sum of F1_c over all C = 5 classes; accuracy = correct / all
- class weight w_c = N / (C x N_c) on the training side (equals scikit-learn `class_weight='balanced'`)
- all-Normal baseline: macro-F1 = 2 pi / (C (1 + pi)), accuracy = pi, where pi is the Normal share of the test set

**Decisions:** a class with no true frames has recall shown as blank (NaN), not 0; a class never predicted has precision blank; F1 of an undefined class counts as 0, so macro-F1 always averages over all 5 classes (same as scikit-learn with zero_division=0).

**Check against the tracker:** the tests confirm that pi = 0.9944 and C = 5 give 0.1994. Note that 0.9944 is the CICIoV2024 share from the base paper; the HCRL Normal share will be different, so every real run prints its own baseline.

**Functions (LIVE, read from the source):**

| Function | What it does |
|---|---|
| confusion_matrix |  |
| per_class | Per-class precision, recall, F1 and support from a confusion matrix. |
| macro_f1 |  |
| accuracy |  |
| trivial_baseline_macro_f1 | Macro-F1 of 'always predict Normal': 2*pi / (C * (1 + pi)). Accuracy = pi. |
| class_weights | w_c = N / (C * N_c) on the training side (proposal Step 6). |
| evaluate | One flat dict of everything a results row needs. |
| all_normal_baseline | Empirical all-Normal predictor on the given labels (should match the formula). |

### M2 - Models and config (`src/models.py`, `config.yaml`, `tests/test_models.py`)

**What:** the three models, built from one place with settings that come only from `config.yaml`.

- **LR:** StandardScaler + LogisticRegression, inside one pipeline, so the scaler is fitted on the training rows only.
- **DT:** DecisionTreeClassifier on the raw integer columns, no depth limit.
- **RF:** RandomForestClassifier on the raw integer columns, bootstrap on, `max_samples` set from the M3 test.
- All three: `class_weight='balanced'`, `random_state` = the run's seed (42 to 46). Same settings for P1 and P2.

**Safety flags in config.yaml:** `frozen` (RF settings written down) and `validity.tolerance_fixed` (supervisor decision at S3). The runner refuses to touch real data until both are true.

**Function (LIVE):**

| Function | What it does |
|---|---|
| build_model |  |

### M3 - Timing and memory test (`src/timing_test.py`)

**What:** shows whether LR, DT and RF can be trained on about 14 million rows with the RAM and time available, without looking at any real label.

**How it works:** it generates synthetic CAN-like data (10 columns, 5 classes), fits each model in its own subprocess at each size (so memory numbers are clean), records fit time, prediction time, peak RAM and tree size, extrapolates to the target size (time: log-log fit; RAM: linear fit), and tests Random Forest candidates (n_estimators, max_samples) against a time budget (60 min per fit) and a RAM budget. Results are written to new files and never overwrite old ones.

**Command used:** `python -m src.timing_test --sizes 1000000 4000000 --target 14000000 --ram-gb 9 --cores 2 --budget-min 60`

**Limits (say these in the report):** synthetic data only has the right shape, so real tree size and time may differ; estimates are rough and optimistic. The first real RF run gives the true time.


### M4 - Experiment runner (`src/run_experiments.py`, `tests/test_runner.py`)

**What:** runs every (model, protocol, seed) combination and appends one row per run to `results/runs.csv`.

**Input assumptions (to be confirmed with the teammates):** `data/frames.parquet` with columns id, dlc, d0 to d7, label (names Normal/DoS/Fuzzy/RPM/Gear or 0 to 4); split files `splits/P1_seed42.npy` and `splits/P2_seed42.npy` (and 43 to 46) holding the TEST row positions; training rows are all other rows.

**Behaviour:**
- One row per run with date, model, protocol, seed, status, validity, accuracy, macro-F1, per-class precision, recall and F1, n_train, n_test, Normal share of the test set, the all-Normal baseline, fit and predict seconds, missing classes, config hash, note.
- A safety-net validity check per split: every class present in train and test and inside the test-share tolerance; an invalid split is recorded as `skipped_invalid` and not trained (proposal: invalid splits are not forced into the comparison).
- Resume: finished runs are skipped when the same command is started again; a crashed run is stored as `error` and retried.
- The results file is only appended to, never rewritten; one confusion matrix per run is saved in `results/confusion/`.
- Guards: a real run is refused until `frozen` and `tolerance_fixed` are true; dry runs cannot write to `runs.csv`.
- Convergence warnings (for example from LR) are written into the `note` column.

**Functions (LIVE):**

| Function | What it does |
|---|---|
| check_split | Safety net only. Raahim's validity_report.csv stays the official record. |
| load_test_idx |  |
| _finished_keys |  |
| _append |  |
| run_experiments |  |
| main |  |

## 5. Fixed configuration (LIVE, `config.yaml`)

```yaml
# Fixed experiment settings (task M2). Written down BEFORE any real result is seen.
# Same settings for P1 and P2. Change nothing here after the first real run.
version: 1

# The runner refuses to touch real data until BOTH flags below are true.
frozen: true             # RF settings fixed after M3, 28 Oct 2026
validity:
  tolerance_fixed: false   # set to true after the supervisor meeting S3 (Thu 29 Oct)
  min_test_share: 0.10     # provisional; per class, test frames / all frames of the class
  max_test_share: 0.30
  on_invalid: skip         # skip = record the split as invalid and do not train (proposal Step 3)
                           # run  = train anyway (only after the team agrees a rule at S3)

classes: [Normal, DoS, Fuzzy, RPM, Gear]
seeds: [42, 43, 44, 45, 46]
split_files: "splits/{protocol}_seed{seed}.npy"   # test-row indices into frames.parquet; CONFIRM with Raahim

models:
  LR:
    C: 1.0
    solver: lbfgs
    max_iter: 200
    tol: 0.0001
    class_weight: balanced
  DT:
    criterion: gini
    max_depth: null
    min_samples_leaf: 1
    class_weight: balanced
  RF:
    n_estimators: 100      # fixed after M3 timing test, 28 Oct 2026
    max_samples: 0.5       # fixed after M3 timing test, 28 Oct 2026; same for P1 and P2
    max_depth: null
    min_samples_leaf: 1
    bootstrap: true
    class_weight: balanced

runtime:                   # hardware only, does not change results
  n_jobs: -1
```

SHA-256 of config.yaml (first 12 characters): `e9a62c75c94a` (the runner stores its own hash of the model section and seeds in every result row).

## 6. Decision on the Random Forest settings

The timing test recommended the first candidate inside the budgets. The team member chose one row lower: **n_estimators = 100, max_samples = 0.5**, because the estimate for full-size samples is above half of the 60-minute budget, estimates from synthetic data are optimistic, and 10 RF runs are needed (2 protocols x 5 seeds). The settings are identical for P1 and P2 and were fixed before any real result existed. The report must state that each tree sees half of the training rows.

## 7. Timing test results (LIVE)

File: `results/timing_recommendation.md` (last modified 2026-10-28 08:08)

#### Timing and memory test (2026-10-28 08:08)

Machine: cores=2, RAM total 12.7 GB, RAM budget 9.0 GB. Synthetic data, sizes 1,000,000, 4,000,000 rows x 10 columns. Extrapolated to 14,000,000 training rows (time: log-log fit, RAM: linear fit; rough).

| model | est. fit time (1 core) | est. peak RAM |
|---|---|---|
| LR | 2.1 min (1 core) | 3.6 GB |
| DT | 3.3 min (1 core) | 1.2 GB |
| RF | 76.0 min (100 full trees, 1 core) | 2.1 GB |

##### Random Forest candidates (budget 60 min per fit, 9 GB RAM)

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


Raw measurements: `results/timing_table.csv`

| model | n_rows | n_estimators | fit_s | predict_s | rss_peak_fit_mb | nodes_per_tree |
|---|---|---|---|---|---|---|
| LR | 1000000 |  | 8.15 | 0.111 | 441.9 | 0.0 |
| LR | 4000000 |  | 34.102 | 0.107 | 1199.4 | 0.0 |
| DT | 1000000 |  | 8.54 | 0.109 | 262.1 | 9389.0 |
| DT | 4000000 |  | 44.225 | 0.104 | 486.3 | 14081.0 |
| RF | 1000000 | 8 | 15.781 | 0.63 | 305.2 | 10564.0 |
| RF | 4000000 | 8 | 82.164 | 0.699 | 720.0 | 15082.0 |

Other timing files present (earlier runs are kept, never overwritten): `timing_extrapolation.csv`


## 8. Test results (LIVE, executed now)

Summary: **17 passed in 3.00s**


`tests/test_metrics.py`

| Test | Result |
|---|---|
| test_confusion_matrix_matches_sklearn | PASSED |
| test_per_class_and_macro_match_sklearn | PASSED |
| test_perfect_prediction | PASSED |
| test_all_normal_baseline_matches_proposal_number | PASSED |
| test_all_normal_baseline_formula_equals_empirical | PASSED |
| test_undefined_cases_are_explicit | PASSED |
| test_class_weights_match_sklearn_balanced | PASSED |
| test_bad_labels_rejected | PASSED |

`tests/test_models.py`

| Test | Result |
|---|---|
| test_builders_and_seeds | PASSED |
| test_scaler_fitted_on_training_side_only | PASSED |
| test_same_seed_same_predictions | PASSED |
| test_unknown_model_rejected | PASSED |

`tests/test_runner.py`

| Test | Result |
|---|---|
| test_dry_run_columns_resume_and_invalid_handling | PASSED |
| test_invalid_split_is_recorded_not_forced | PASSED |
| test_check_split_counts | PASSED |
| test_real_run_refused_until_frozen_and_tolerance_fixed | PASSED |
| test_label_strings_are_mapped | PASSED |


## 9. Runner dry run (LIVE, synthetic data only, executed now)

A small dry run on 20,000 synthetic rows. **These numbers mean nothing for the project**, they only prove that the code works, writes the expected columns and resumes correctly.

| model | protocol | seed | status | valid | accuracy | macro_f1 | baseline_macro_f1 | fit_seconds | n_train | n_test | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LR | P1 | 42 | ok | True | 0.5427 | 0.4082 | 0.1841 | 0.2300 | 16000 | 4000 | DeprecationWarning |
| DT | P1 | 42 | ok | True | 0.9647 | 0.8766 | 0.1841 | 0.1200 | 16000 | 4000 |  |
| RF | P1 | 42 | ok | True | 0.9640 | 0.8209 | 0.1841 | 1.6400 | 16000 | 4000 |  |
| LR | P2 | 42 | ok | True | 0.4713 | 0.3955 | 0.1865 | 0.8800 | 15994 | 4006 | DeprecationWarning |
| DT | P2 | 42 | ok | True | 0.7806 | 0.7907 | 0.1865 | 0.1800 | 15994 | 4006 |  |
| RF | P2 | 42 | ok | True | 0.9596 | 0.7949 | 0.1865 | 2.2000 | 15994 | 4006 |  |

Columns in the results file (33): `date`, `model`, `protocol`, `seed`, `status`, `valid`, `accuracy`, `macro_f1`, `recall_Normal`, `recall_DoS`, `recall_Fuzzy`, `recall_RPM`, `recall_Gear`, `precision_Normal`, `precision_DoS`, `precision_Fuzzy`, `precision_RPM`, `precision_Gear`, `f1_Normal`, `f1_DoS`, `f1_Fuzzy`, `f1_RPM`, `f1_Gear`, `n_train`, `n_test`, `pi_N_test`, `baseline_macro_f1`, `fit_seconds`, `predict_seconds`, `missing_train`, `missing_test`, `config_hash`, `note`


Second start of the same command (resume check):

```
DRY RUN on 20,000 synthetic rows -> /tmp/doc_dryrun_gebhoq1d/dry.csv
[skip] P1 seed 42: all models already in dry.csv
[skip] P2 seed 42: all models already in dry.csv
```

Guard check: starting a REAL run with the current config gives exit code `2`:

```
Refusing to run on real data:
  - config.yaml: validity.tolerance_fixed is false (decide at S3, Thu 29 Oct)
```


## 10. Rules followed and known limits

| Rule or limit | How it is handled |
|---|---|
| No real data before the tolerance is fixed (ground rule 3) | Runner refuses real runs until `tolerance_fixed` is true; all M1 to M4 work used synthetic data only |
| Hyperparameters fixed before results (ground rule 4) | `frozen: true` written after the timing test and before any real result; same values for P1 and P2 |
| Results files are never overwritten (ground rule 6) | Timing files get a timestamp if a file exists; runner only appends; confusion matrices are not overwritten |
| Check results in order (ground rule 8) | Procedure in docs/SAMEER_GUIDE.md, Step 9 |
| Timing estimates | From synthetic data, rough and optimistic; re-check with the real number of signatures and on the first real RF run |
| Input formats | Assumed (see M4); to be confirmed with the teammates who produce frames.parquet and the split files |
| LR convergence on 14M real rows | Not tested on real data; a ConvergenceWarning is recorded in the `note` column and must be reported, not hidden by changing max_iter afterwards |
| Dummy splits in src/synthetic.py | Only for dry runs; the official splits come from the splits track |

## 11. Status and sign-off

| Task | Built | Tested | Teammate check (name, date) | Status |
|---|---|---|---|---|
| M1 metrics | yes | yes (see section 8) |  | Done once checked |
| M2 models and config | yes | yes (see section 8) |  | Done once checked |
| M3 timing test | yes | yes, real run done 28 Oct 2026 |  | Done once checked |
| M4 runner | yes | yes, dry run and resume (section 9) |  | Done once checked |
| M5 first real run | - | - | - | Blocked: frames.parquet, splits, S3 tolerance |

What is needed next: `data/frames.parquet`, the split files, the agreed tolerance (set `tolerance_fixed: true` and the min and max test shares in `config.yaml`), then:

```
python -m src.run_experiments --seeds 42 --models LR DT
```

## 12. How to reproduce everything

```
%cd /content/drive/MyDrive/FYDP/files/fydp_models_sameer/fydp_models   # Colab
pip install -r requirements.txt
python -m pytest -v
python -m src.timing_test --sizes 1000000 4000000 --target 14000000 --ram-gb 9 --cores 2 --budget-min 60
python -m src.run_experiments --synthetic 200000 --models LR DT RF --seeds 42
python -m src.make_documentation
```
