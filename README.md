# FYDP: Data redundancy and train-test splitting on the HCRL Car-Hacking dataset

## Setup
    python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Data (not in Git)
Put the five raw files in `data/raw/`:
normal_run_data.txt, DoS_dataset.csv, Fuzzy_dataset.csv, RPM_dataset.csv, gear_dataset.csv
Then record checksums:  sha256sum data/raw/* > data/raw/CHECKSUMS.txt  (commit CHECKSUMS.txt)

## Pipeline
    python src/build_dataset.py      # raw files -> data/processed/frames.parquet
    python src/make_sample.py        # -> data/sample/frames_sample.parquet (for code testing only)

## Frame table schema (one row per CAN frame)
ts float64 | id uint16 | dlc uint8 | d0..d7 uint8 (0 beyond DLC) | data64 uint64 (8 bytes packed)
label {Normal,DoS,Fuzzy,RPM,Gear} | source {normal,dos,fuzzy,rpm,gear} | flag {R,T} | sig_id int32
Signature = (id, dlc, d0..d7); sig_id is a unique integer per distinct signature (exact, no hash collisions).

## Rules
- The sample is for testing code only. Never report a number computed on it.
- Seeds: 42, 43, 44, 45, 46. Decisions are logged in DECISIONS.md with a date.
