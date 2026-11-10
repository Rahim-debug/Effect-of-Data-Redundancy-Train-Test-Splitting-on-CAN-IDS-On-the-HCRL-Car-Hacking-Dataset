# Sameer's guide: Sprint 1 models track (M1 to M5)

Written on Tue 27 Oct 2026. The plan started Thu 22 Oct, so M1 and M2 are already due
(Mon 26 / Tue 27). All code for M1 to M4 is in this folder and passes its tests; your job
is to run it, check it, commit it, and get the numbers only your machine can give.
M5 cannot be done yet: it needs Hafiz's `frames.parquet`, Raahim's split files and the
supervisor decision (S3).

## Catch-up schedule (replaces the Day plan for Sameer)

| Day | Do | Tasks |
|---|---|---|
| Tue 27 | Steps 1 to 4 below, send Message A and B | M1 done, M2 done, M3 started |
| Wed 28 | Steps 5 to 7: timing test, fix RF settings, set `frozen: true` | M3 done, M4 dry run |
| Thu 29 | Step 8, attend S3, set `tolerance_fixed: true` | M4 done |
| Fri 30 | Step 9 | M5, then Weekly Report 10 |

## Step 1. Put the files in the repo (S1)

1. Clone the team repo. Copy `src/`, `tests/`, `config.yaml`, `requirements.txt`, `conftest.py` into it
   (if the repo already has a `conftest.py` or `requirements.txt`, merge them, do not overwrite).
2. `pip install -r requirements.txt`
3. Open `src/common.py`. Its class order, 10 feature names, seeds and 80/20 must equal what
   `CONVENTIONS.md` says. If they differ, CONVENTIONS.md wins: edit `common.py`.
4. Commit: `git add src tests config.yaml requirements.txt && git commit -m "M1-M4: metrics, models, timing, runner"`.
   Never commit `data/`.

## Step 2. M1: check the metrics (about 15 min)

Run `pytest tests/test_metrics.py -v`. All 8 tests must pass. They show that your numbers equal scikit-learn's
(confusion matrix, per-class precision, recall, F1, macro-F1, accuracy, class weights) and that the all-Normal
baseline formula gives 0.1994 for pi_N = 0.9944, C = 5, which is the "Done when" in the tracker.

Read `src/metrics.py` once, you will have to explain it. Points to know:
- Rows of the confusion matrix are the true class, columns are the predicted class.
- A class with no true frames gets recall = blank (NaN), not 0. F1 of an undefined class counts as 0 in macro-F1.
- Warning: 0.9944 is the CICIoV2024 share from the base paper. The HCRL Normal share will be different.
  The tests use 0.9944 only to check the formula; every real run prints its own baseline from the real test set.

Ask Hafiz or Raahim to read the tests and say "checked". Then mark M1 Done.

## Step 3. M2: check the models and the config (about 20 min)

Run `pytest tests/test_models.py -v` (4 tests). They show: seed = random_state, `class_weight='balanced'`
for all three (= the proposal's w_c = N / (C N_c) on the training side), and that LR's scaler is fitted on the
training rows only.

Open `config.yaml`. Everything is fixed except two PROVISIONAL RF values (`n_estimators`, `max_samples`),
which Step 6 settles. Rule 4: after the first real result, nothing in `models:` changes.
Mark M2 "waiting for RF settings" until Step 6.

## Step 4. Machine specs and interface check (Decision #4 was due Fri 23, it is late)

Windows: Task Manager > Performance (RAM, cores). Linux: `free -g` and `nproc`.
Send these two messages today:

Message A (team chat): "RAM and cores of the machine for the heavy runs: ___ GB, ___ cores. Which machine is it?"

Message B (Hafiz and Raahim): "My runner assumes: (1) data/frames.parquet has columns id, dlc, d0-d7, label, with
label = Normal/DoS/Fuzzy/RPM/Gear (or 0-4 in that order); (2) split files are splits/P1_seed42.npy and
splits/P2_seed42.npy (also 43-46) holding the TEST row positions 0..N-1 in the row order of frames.parquet;
train is everything else. Is that what you are producing? If not, tell me the real names/format today."
If they say no, edit `split_files:` in `config.yaml` (and `load_test_idx` in `src/run_experiments.py` if needed).

## Step 5. M3: timing and memory test (start today, let it run)

Dry check first (1 minute): `python -m src.timing_test --sizes 20000 60000 --out-dir /tmp/t`
Then the real test on the team machine:

    python -m src.timing_test --sizes 1000000 4000000 --target 14000000 --ram-gb <RAM you can use> --cores <cores> --budget-min 60

It trains LR, DT and RF on SYNTHETIC data (no real labels, so rule 3 is respected), measures time and peak RAM, and
extrapolates to 14M training rows. Expect 10 to 40 minutes. It writes `results/timing_table.csv`,
`results/timing_extrapolation.csv` and `results/timing_recommendation.md` (never overwrites: a new timestamped file
is made if one exists). Close other heavy programs while it runs.

How to read it: the recommendation file lists RF candidates (n_estimators, max_samples) with estimated time and RAM
and picks the first one inside your budgets. Treat the estimates as optimistic: synthetic trees are small, real tree
size depends on the real number of signatures. Keep a safety margin (if the estimate is above half your budget, move
one row down the list). The final check is Step 9: the first real RF run shows the true time.

## Step 6. Fix the settings (Wed 28 morning)

1. Copy the recommended `n_estimators` and `max_samples` into `config.yaml` (RF section).
2. Look at the LR line in `timing_recommendation.md`. If LR is too slow or may not converge at 14M rows, decide NOW
   (for example a larger `max_iter`), before any real result exists.
3. Set `frozen: true`. Commit with message "M2/M3: hyperparameters frozen". Mark M2 and M3 Done after a teammate checks them.

## Step 7. M4: dry-run the runner (Wed 28)

    pytest tests/test_runner.py -v
    python -m src.run_experiments --synthetic 200000 --models LR DT RF --seeds 42

Check: one row per run in `results/dry_run_runs.csv`, all columns filled, a `[skip]` message when you run the same
command a second time (that is the resume after a crash), and `[invalid]` plus status `skipped_invalid` if a split
has a class missing on one side (invalid splits are recorded, not forced, as the proposal says).
The synthetic numbers mean NOTHING for the project, they only prove the code works. Delete `dry_run_runs.csv` after.

What the runner enforces by itself:
- Real runs are refused until `frozen: true` AND `validity.tolerance_fixed: true` (rules 3 and 4).
- `results/runs.csv` is only appended to. Each row has date, seed, protocol, config hash. A confusion matrix is saved per run.
- A run that crashes is stored as `error` and retried next time you start the same command.

## Step 8. Thursday: S3 meeting, then unlock

At S3 bring: `timing_recommendation.md`, Raahim's `validity_report.csv`, Hafiz's signatures-per-class numbers.
After the tolerance is agreed: put it in `config.yaml` (`min_test_share`, `max_test_share`), set `tolerance_fixed: true`
and, if the team agrees a rule for invalid classes, set `on_invalid` (default `skip`). Commit.
Mark M4 Done once a teammate has run your dry run on their laptop.

## Step 9. M5: first real run (Fri 30, only after S3)

    python -m src.run_experiments --frames data/frames.parquet --seeds 42 --models LR DT
    python -m src.run_experiments --frames data/frames.parquet --seeds 42 --models RF      # only if time allows

Then open `results/runs.csv` and check, in this order (rule 8):
1. `status` is `ok` and `valid` is True for both protocols. A `skipped_invalid` P2 row is a RESULT (a class that cannot be
   split), not a failure: report it, bring it to the supervisor.
2. `macro_f1` is clearly above `baseline_macro_f1` (the all-Normal score on that test set).
3. If P1 is near-perfect, that is expected from the proposal. If P2 is ALSO near-perfect, or P2 is better than P1
   for a class, check leakage before believing it (the overlap numbers in Raahim's validity report).
4. Read `note`: `ConvergenceWarning` means LR did not converge. Report it, do not quietly change `max_iter` afterwards.
Share the first P1 vs P2 numbers with the team, mark M5 Done (or carry over with a reason), and give Weekly Report 10
only files and numbers that exist in the repo.

## Status lines to paste into the tracker

M1: Done after teammate check | M2: Done after Step 6 | M3: Done after Step 6 | M4: Done after Step 8 |
M5: Blocked on H2 (frames.parquet), R3 (splits), S3 (tolerance).
