"""Task M4 - experiment runner: models x protocols x seeds -> results/runs.csv.

Real run (only after config.yaml has frozen: true AND validity.tolerance_fixed: true):
    python -m src.run_experiments --frames data/frames.parquet --out results/runs.csv

Dry run on synthetic data (safe any time, never touches real data):
    python -m src.run_experiments --synthetic 200000 --models LR DT --seeds 42 43

One row per (model, protocol, seed). The file is only ever appended to, never rewritten
(ground rule 6). After a crash, run the same command again: finished runs are skipped.
Split files hold the TEST row positions (0-based, into frames.parquet row order);
train = every other row.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from .common import (CLASSES, MODELS, N_CLASSES, PROTOCOLS, config_hash, load_config, load_frames)
from .metrics import all_normal_baseline, confusion_matrix, evaluate
from .models import build_model
from .synthetic import make_synthetic_arrays, write_dummy_splits

ROOT = Path(__file__).resolve().parents[1]
RUN_COLUMNS = (
    ["date", "model", "protocol", "seed", "status", "valid", "accuracy", "macro_f1"]
    + [f"recall_{c}" for c in CLASSES] + [f"precision_{c}" for c in CLASSES] + [f"f1_{c}" for c in CLASSES]
    + ["n_train", "n_test", "pi_N_test", "baseline_macro_f1", "fit_seconds", "predict_seconds",
       "missing_train", "missing_test", "config_hash", "note"]
)
FINISHED = {"ok", "skipped_invalid"}


def check_split(y, test_idx, validity_cfg) -> dict:
    """Safety net only. Raahim's validity_report.csv stays the official record."""
    test_mask = np.zeros(len(y), dtype=bool); test_mask[test_idx] = True
    n_tr = np.bincount(y[~test_mask], minlength=N_CLASSES)
    n_te = np.bincount(y[test_mask], minlength=N_CLASSES)
    miss_tr = [CLASSES[i] for i in range(N_CLASSES) if n_tr[i] == 0]
    miss_te = [CLASSES[i] for i in range(N_CLASSES) if n_te[i] == 0]
    tot = n_tr + n_te
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(tot > 0, n_te / tot, np.nan)
    lo, hi = validity_cfg["min_test_share"], validity_cfg["max_test_share"]
    bad = [CLASSES[i] for i in range(N_CLASSES) if tot[i] > 0 and not (lo <= share[i] <= hi)]
    return {"valid": not miss_tr and not miss_te and not bad, "missing_train": miss_tr,
            "missing_test": miss_te, "out_of_tolerance": bad, "test_mask": test_mask}


def load_test_idx(path: Path, n_rows: int) -> np.ndarray:
    idx = np.load(path)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError(f"{path}: expected a 1-D non-empty array of test row positions")
    if idx.min() < 0 or idx.max() >= n_rows:
        raise ValueError(f"{path}: indices outside 0..{n_rows - 1}; are these positions in frames.parquet?")
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{path}: duplicate indices")
    return idx.astype(np.int64)


def _finished_keys(out: Path) -> set:
    if not out.exists() or out.stat().st_size == 0:
        return set()
    with open(out, newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != RUN_COLUMNS:
            raise SystemExit(f"{out} has different columns from this runner; write to a NEW file instead.")
        return {(r["model"], r["protocol"], int(r["seed"]), r["config_hash"]) for r in rd if r["status"] in FINISHED}


def _append(out: Path, row: dict):
    new = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUN_COLUMNS})


def run_experiments(X, y, cfg, splits_dir, out, models, protocols, seeds, log=print) -> int:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    chash = config_hash(cfg)
    done = _finished_keys(out)
    n_new = 0
    for seed in seeds:
        for proto in protocols:
            todo = [m for m in models if (m, proto, seed, chash) not in done]
            if not todo:
                log(f"[skip] {proto} seed {seed}: all models already in {out.name}")
                continue
            path = Path(splits_dir) / cfg["split_files"].split("/")[-1].format(protocol=proto, seed=seed)
            test_idx = load_test_idx(path, len(y))
            chk = check_split(y, test_idx, cfg["validity"])
            test_mask = chk.pop("test_mask")
            base = {"protocol": proto, "seed": seed, "config_hash": chash, "valid": chk["valid"],
                    "missing_train": ";".join(chk["missing_train"]), "missing_test": ";".join(chk["missing_test"]),
                    "n_train": int((~test_mask).sum()), "n_test": int(test_mask.sum())}
            if not chk["valid"] and cfg["validity"]["on_invalid"] == "skip":
                note = "invalid split: " + (f"missing in train {chk['missing_train']} " if chk["missing_train"] else "") \
                    + (f"missing in test {chk['missing_test']} " if chk["missing_test"] else "") \
                    + (f"test share outside tolerance {chk['out_of_tolerance']}" if chk["out_of_tolerance"] else "")
                for m in todo:
                    _append(out, {**base, "model": m, "status": "skipped_invalid", "note": note.strip(),
                                  "date": datetime.now().isoformat(timespec="seconds")})
                log(f"[invalid] {proto} seed {seed}: {note.strip()}  (not trained)")
                n_new += len(todo)
                continue
            Xtr, ytr, Xte, yte = X[~test_mask], y[~test_mask], X[test_mask], y[test_mask]
            bl = all_normal_baseline(yte)
            for m in todo:
                row = {**base, "model": m, "pi_N_test": bl["pi_normal"], "baseline_macro_f1": bl["macro_f1"]}
                try:
                    model = build_model(m, seed, cfg)
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
                    t0 = time.perf_counter(); pred = model.predict(Xte); pred_s = time.perf_counter() - t0
                    ev = evaluate(yte, pred)
                    row.update({k: v for k, v in ev.items() if k in RUN_COLUMNS})
                    row.update(status="ok", fit_seconds=round(fit_s, 2), predict_seconds=round(pred_s, 2),
                               note="; ".join(sorted({str(x.category.__name__) for x in w})))
                    cm_dir = out.parent / "confusion"; cm_dir.mkdir(exist_ok=True)
                    cm_path = cm_dir / f"{out.stem}_{m}_{proto}_seed{seed}.npy"
                    if not cm_path.exists():
                        np.save(cm_path, confusion_matrix(yte, pred))
                    log(f"[ok] {m} {proto} seed {seed}: macro-F1 {ev['macro_f1']:.4f}, acc {ev['accuracy']:.4f} "
                        f"(all-Normal baseline macro-F1 {bl['macro_f1']:.4f}), fit {fit_s:.0f}s")
                    del model
                except Exception as e:  # keep going; this run is retried on the next start
                    row.update(status="error", note=f"{type(e).__name__}: {e}"[:300])
                    log(f"[error] {m} {proto} seed {seed}: {row['note']}")
                row["date"] = datetime.now().isoformat(timespec="seconds")
                _append(out, row); n_new += 1
    return n_new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames.parquet"))
    ap.add_argument("--splits-dir", default=str(ROOT / "splits"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS, choices=PROTOCOLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--synthetic", type=int, default=0, metavar="N", help="dry run on N synthetic rows")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    seeds = a.seeds or cfg["seeds"]

    if a.synthetic:                                   # ---- dry run
        out = Path(a.out or ROOT / "results" / "dry_run_runs.csv")
        if out.name == "runs.csv":
            raise SystemExit("dry runs must not write to runs.csv; choose another --out")
        X, y = make_synthetic_arrays(a.synthetic, seed=0)
        tmp = tempfile.mkdtemp(prefix="dummy_splits_")
        write_dummy_splits(X, y, seeds, tmp)
        cfg["validity"]["on_invalid"] = cfg["validity"].get("on_invalid", "skip")
        print(f"DRY RUN on {a.synthetic:,} synthetic rows -> {out}")
        return 0 if run_experiments(X, y, cfg, tmp, out, a.models, a.protocols, seeds) >= 0 else 1

    # ---- real run: ground rules 3 and 4 enforced in code
    problems = []
    if not cfg.get("frozen"):
        problems.append("config.yaml: frozen is false (fix RF settings from M3 and write them down first)")
    if not cfg["validity"].get("tolerance_fixed"):
        problems.append("config.yaml: validity.tolerance_fixed is false (decide at S3, Thu 29 Oct)")
    if problems:
        print("Refusing to run on real data:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 2
    out = Path(a.out or ROOT / "results" / "runs.csv")
    X, y = load_frames(a.frames)
    print(f"loaded {len(y):,} frames; class counts {dict(zip(CLASSES, np.bincount(y, minlength=N_CLASSES).tolist()))}")
    run_experiments(X, y, cfg, a.splits_dir, out, a.models, a.protocols, seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
