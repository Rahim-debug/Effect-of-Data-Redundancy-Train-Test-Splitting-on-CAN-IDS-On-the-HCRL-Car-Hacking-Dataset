# Verification Report

## 1 Environment

Required packages in `requirements.txt`: pandas, numpy, pyarrow, scikit-learn, and matplotlib are present; pytest is missing.

```text
--- python --version ---
Python 3.13.1
--- pip --version ---
pip 24.3.1 from D:\Free Download Manager\Study\7th Sem\Final year design project\Research Paper\EV Security\Cluade\files\fydp-can-ids\fydp-can-ids\.venv\Lib\site-packages\pip (python 3.13)
--- active python ---
D:\Free Download Manager\Study\7th Sem\Final year design project\Research Paper\EV Security\Cluade\files\fydp-can-ids\fydp-can-ids\.venv\Scripts\python.exe
--- requirements check ---
pandas: present
numpy: present
pyarrow: present
scikit-learn: present
pytest: MISSING
matplotlib: present
--- package versions ---
3.1.0 2.5.4 26.0.0 1.9.1
--- total RAM GB ---
15.9
```

`pip install -r requirements.txt` completed with exit code 0. The installed packages listed in `requirements.txt` were already satisfied.

## 2 Repo state

```text
--- git status --short ---
 M DECISIONS.md
?? results/day1_sanity1.txt
?? src/audit.py
?? src/checks.py
?? src/gaps.py
?? src/parse.py
?? src/pilot.py
?? src/random_overlap.py
?? src/report_audit.py
?? src/report_data.py
?? src/run_technique_b.py
?? src/sanitise.py
?? src/splits.py
?? src/temporal_audit.py
?? src/test_splits.py
--- branch ---
main
--- git count-objects -vH ---
count: 80
size: 6.08 MiB
in-pack: 0
packs: 0
size-pack: 0 bytes
prune-packable: 0
garbage: 0
size-garbage: 0 bytes
--- .gitignore ---
# raw and processed data are large: never commit them (data/sample IS committed)
data/raw/
data/processed/
__pycache__/
.ipynb_checkpoints/
.venv/
*.pyc
--- .gitignore entry checks ---
.venv/ : present
__pycache__/ : present
data/ : MISSING
*.parquet : MISSING
--- git diff --stat DECISIONS.md ---
warning: in the working copy of 'DECISIONS.md', LF will be replaced by CRLF the next time Git touches it
 DECISIONS.md | 75 ++++++++++++++++++++++++++++++++++++++++++++----------------
 1 file changed, 55 insertions(+), 20 deletions(-)
--- files in src/, tests/, data/ (recursive; sizes in bytes) ---
src\audit.py`t2406
src\build_dataset.py`t1782
src\checks.py`t2896
src\config.py`t369
src\gaps.py`t1435
src\make_sample.py`t838
src\parse.py`t4870
src\parse_raw.py`t3118
src\pilot.py`t4459
src\random_overlap.py`t1518
src\report_audit.py`t7440
src\report_data.py`t9104
src\run_technique_b.py`t3981
src\sanitise.py`t1409
src\splits.py`t8983
src\temporal_audit.py`t3660
src\test_splits.py`t9856
tests\make_fake_raw.py`t1325
data\processed\frames.parquet`t226980830
data\raw\CHECKSUMS.txt`t508
data\raw\DoS_dataset.csv`t190058708
data\raw\Fuzzy_dataset.csv`t198520767
data\raw\gear_dataset.csv`t230320414
data\raw\normal_run_data.7z`t7510943
data\raw\normal_run_data.txt`t87371743
data\raw\normal_run_data.txt.bak`t87371776
data\raw\normal_run_data\normal_run_data.txt`t87371776
data\raw\RPM_dataset.csv`t239581936
data\sample\frames_sample.parquet`t7742587
--- files in results/ (recursive; sizes in bytes) ---
results\day1_sanity.txt`t2828
results\day1_sanity1.txt`t2832
```

## 3 File placement

```text
Moved src/test_splits.py to tests/test_splits.py
Created empty src/__init__.py
tests\test_splits.py 9856 bytes
src\__init__.py 0 bytes
```

## 4 Import check

```text
imports ok
```

## 5 Data check

`data/derived/frames_clean.parquet` exists. Its metadata row count is 13,994,074 and its size is 138.100565910339 MB. `data/derived/frames.parquet` was not found. The earlier inventory also listed:

```text
Size MB: 138.100565910339
13994074
```

## 6 Tests

Command: `python -m pytest -v tests`

```text
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.2.0, pluggy-1.7.0 -- D:\Free Download Manager\Study\7th Sem\Final year design project\Research Paper\EV Security\Cluade\files\fydp-can-ids\fydp-can-ids\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Free Download Manager\Study\7th Sem\Final year design project\Research Paper\EV Security\Cluade\files\fydp-can-ids\fydp-can-ids
plugins: anyio-4.16.0, platformdirs-2.8.0
collecting ... collected 22 items

tests/test_splits.py::test_a_exact_count_partition_sorted[False] PASSED  [  4%]
tests/test_splits.py::test_a_exact_count_partition_sorted[True] PASSED   [  9%]
tests/test_splits.py::test_a_stratified_exact_per_class PASSED           [ 13%]
tests/test_splits.py::test_a_plain_shuffle_keeps_every_class_about_20_percent PASSED [ 18%]
tests/test_splits.py::test_a_seed_controls_the_draw[False] PASSED        [ 22%]
tests/test_splits.py::test_a_seed_controls_the_draw[True] PASSED         [ 27%]
tests/test_splits.py::test_a_is_not_chronological PASSED                 [ 31%]
tests/test_splits.py::test_b_train_entirely_before_test_in_every_file PASSED [ 36%]
tests/test_splits.py::test_b_partition_and_deterministic PASSED          [ 40%]
tests/test_splits.py::test_b_never_cuts_a_burst PASSED                   [ 45%]
tests/test_splits.py::test_b_cut_is_in_the_middle_of_a_quiet_stretch PASSED [ 50%]
tests/test_splits.py::test_b_all_classes_on_both_sides_and_shares_in_range PASSED [ 54%]
tests/test_splits.py::test_b_ambient_file_cut_at_80_percent_of_duration PASSED [ 59%]
tests/test_splits.py::test_b_files_are_independent_of_absolute_time PASSED [ 63%]
tests/test_splits.py::test_b_80_percent_point_inside_a_burst_is_moved_to_the_next_quiet_stretch PASSED [ 68%]
tests/test_splits.py::test_b_80_percent_point_in_a_quiet_stretch_uses_that_stretch PASSED [ 72%]
tests/test_splits.py::test_b_warns_when_the_table_looks_unsanitised PASSED [ 77%]
tests/test_splits.py::test_b_no_warning_on_a_sanitised_table PASSED      [ 81%]
tests/test_splits.py::test_b_without_sanitisation_test_side_has_no_attacks PASSED [ 86%]
tests/test_splits.py::test_seen_share_single_signature_vs_unique_signature PASSED [ 90%]
tests/test_splits.py::test_group_split_diagnostic_still_clean PASSED     [ 95%]
tests/test_splits.py::test_bad_inputs PASSED                             [100%]

============================= 22 passed in 2.00s ==============================
```

## 7 Technique B run

Parquet metadata row count and file size:

```text
Size MB: 138.100565910339
13994074
```

Command:
`python src/run_technique_b.py --parquet data/derived/frames_clean.parquet --out results --splits-dir data/splits 2>&1 | Tee-Object -FilePath results\run_log_technique_b.txt`

Raw command output:

```text
loaded 13,994,074 rows in 1.1s

PER FILE
            src_file  n_frames  n_bursts  cut_rel_s  cut_moved_out_of_burst  quiet_at_cut_s  max_gap_s  n_train  n_test  test_share
normal_run_data.txt    988871         0    405.149                   False             NaN      0.006   791099  197772       0.200
    DoS_dataset.csv   2774703       300   1864.503                    True           3.004      0.245  2191554  583149       0.210
  Fuzzy_dataset.csv   2947792       300   1972.794                   False           3.013      0.712  2427081  520711       0.177
    RPM_dataset.csv   3730634       300   1566.145                    True           3.005      0.676  2966515  764119       0.205
   gear_dataset.csv   3552074       300   1562.469                    True           3.004      0.317  2834660  717414       0.202

PER CLASS
  label   train    test  test_share
Normal 9317710 2344847      0.2011
   DoS  478898  108623      0.1849
 Fuzzy  415726   76121      0.1548
   RPM  519813  135084      0.2063
  Gear  478762  118490      0.1984

saved results/split_b_*.csv and data\splits\technique_B_test_idx.npy  (done in 2.8s)

CHECK PASSED: every class's train and test frame count equals Hafiz's temporal audit.
```

`results/split_b_files.csv`:

```text
src_file,n_frames,t0,t_end,n_bursts,max_gap_s,cut_time,cut_rel_s,cut_moved_out_of_burst,quiet_at_cut_s,n_train,n_test,test_share
normal_run_data.txt,988871,1479121434.850202,1479121941.286773,0,0.006451845169067383,1479121839.9994588,405.1492567062378,False,,791099,197772,0.19999777524065324
DoS_dataset.csv,2774703,1478198376.389427,1478200705.061203,300,0.2448418140411377,1478200240.8924813,1864.5030543804169,True,3.0036609172821045,2191554,583149,0.2101662772556198
Fuzzy_dataset.csv,2947792,1478195721.903877,1478198187.937162,300,0.7124688625335693,1478197694.698275,1972.7943980693817,False,3.013159990310669,2427081,520711,0.17664441724517876
RPM_dataset.csv,3730634,1478191030.045114,1478192982.484684,300,0.676051139831543,1478192596.1899724,1566.1448583602905,True,3.0054690837860107,2966515,764119,0.20482282636141738
gear_dataset.csv,3552074,1478193190.056566,1478195139.293625,300,0.3173489570617676,1478194752.5252366,1562.4686706066132,True,3.0039451122283936,2834660,717414,0.20197045444436124
```

`results/split_b_classes.csv`:

```text
label,train,test,test_share
Normal,9317710,2344847,0.20105770972866413
DoS,478898,108623,0.18488360416053212
Fuzzy,415726,76121,0.1547656080041151
RPM,519813,135084,0.20626755046976852
Gear,478762,118490,0.1983919685492891
```

## 8 Temporal audit

Skipped because `data/derived/frames.parquet` is missing. The existing `data/processed/frames.parquet` does not match the specified optional input path.

## 9 Problems encountered

- Step 1: `pytest` is missing from `requirements.txt`. The environment install command completed successfully; it was not run to change the requirements file.
- Step 8: `data/derived/frames.parquet` is missing, so the temporal audit was skipped.
- No command returned a non-zero exit code; no error traceback was produced.
