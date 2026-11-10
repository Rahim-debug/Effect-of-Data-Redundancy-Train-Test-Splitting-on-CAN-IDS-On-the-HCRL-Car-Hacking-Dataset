"""Generates the complete documentation of tasks M1-M4 from the REAL files in this project.

    python -m src.make_documentation

Writes (never overwrites, a timestamp is part of the name):
    docs/M1_M4_documentation_<YYYYMMDD_HHMM>.md     (text, opens in any editor)
    docs/M1_M4_documentation_<YYYYMMDD_HHMM>.html   (open in a browser, Ctrl+P -> Save as PDF)

Everything marked LIVE is read at the moment you run it (tests are executed, the timing
files and config.yaml are read, a small synthetic dry run is executed in a temp folder).
Nothing here touches real data.
"""
from __future__ import annotations

import ast
import csv
import glob
import hashlib
import html
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ helpers
def sh(cmd, timeout=900):
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # pragma: no cover
        return 99, f"{type(e).__name__}: {e}"


def parse_pytest(output: str):
    """-> (per_file {file: [(test, 'PASSED'/'FAILED')]}, summary_line)"""
    per_file = {}
    for m in re.finditer(r"(tests/[\w./]+\.py)::(\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)", output):
        per_file.setdefault(m.group(1), []).append((m.group(2), m.group(3)))
    summary = ""
    for line in output.strip().splitlines()[::-1]:
        if re.search(r"\d+ (passed|failed|error)", line):
            summary = line.strip("= ").strip()
            break
    return per_file, summary


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |")
    return "\n".join(out)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def latest(pattern: str):
    files = sorted(glob.glob(str(ROOT / pattern)), key=os.path.getmtime)
    return Path(files[-1]) if files else None


def inventory():
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts or ".git" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        st = p.stat()
        rows.append((str(p.relative_to(ROOT)), f"{st.st_size/1024:.1f} KB",
                     datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")))
    return rows


def functions_of(path: Path):
    tree = ast.parse(read_text(path))
    rows = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("__"):
            doc = (ast.get_docstring(node) or "").splitlines()
            rows.append((node.name, doc[0] if doc else ""))
    return rows


def env_info():
    info = {"Python": platform.python_version(), "Platform": platform.platform(),
            "CPU cores": os.cpu_count()}
    try:
        import psutil
        info["RAM total"] = f"{psutil.virtual_memory().total / 2**30:.1f} GB"
    except Exception:
        pass
    for lib in ["numpy", "pandas", "sklearn", "yaml", "pyarrow"]:
        try:
            mod = __import__(lib)
            info[lib] = getattr(mod, "__version__", "installed")
        except Exception:
            info[lib] = "not installed"
    return info


# ------------------------------------------------------------------ sections
def build() -> str:
    now = datetime.now()
    L = []
    add = L.append
    add("# FYDP Sprint 1: models track, tasks M1 to M4 - complete documentation")
    add(f"\nGenerated: {now:%Y-%m-%d %H:%M} (machine clock) by `src/make_documentation.py`. "
        "Sections marked **LIVE** were read or executed at that moment.\n")

    # ---- 0 where
    add("## 1. Where everything is saved\n")
    add(f"Project folder (absolute path on this machine):\n\n`{ROOT}`\n")
    add("On Google Drive this is the same folder: open drive.google.com and follow "
        "`FYDP` > `files` > `fydp_models_sameer` > `fydp_models` (the exact path above, after `/MyDrive/`).\n")
    add("**LIVE file inventory** (what exists now, size, last modified):\n")
    add(md_table(["File (relative to project folder)", "Size", "Last modified"], inventory()))
    add("\nWhat each part is for:\n")
    add(md_table(["Location", "Purpose", "Task"], [
        ("src/common.py", "Fixed class order, 10 feature names, seeds, protocol names, config hash, data loader", "shared"),
        ("src/metrics.py", "Evaluation metrics written from the formulas", "M1"),
        ("tests/test_metrics.py", "8 tests: metrics must equal scikit-learn", "M1"),
        ("src/models.py", "Builders for LR, DT, RF", "M2"),
        ("config.yaml", "All hyperparameters and the two safety flags (frozen, tolerance_fixed)", "M2 / M3"),
        ("tests/test_models.py", "4 tests: seeds, class weights, scaler fitted on training side only", "M2"),
        ("src/timing_test.py", "Timing and RAM test on synthetic data, extrapolation, RF recommendation", "M3"),
        ("results/timing_table.csv, timing_extrapolation.csv, timing_recommendation.md", "Output of the real timing test", "M3"),
        ("src/run_experiments.py", "Experiment runner: models x protocols x seeds -> results/runs.csv", "M4"),
        ("tests/test_runner.py", "5 tests: columns, resume, invalid splits, real-run guard, label mapping", "M4"),
        ("src/synthetic.py", "Synthetic CAN-like data and DUMMY splits, only for tests and dry runs", "M3 / M4"),
        ("results/runs.csv (future)", "Real results, appended by the runner (does not exist yet)", "M5"),
        ("docs/", "This documentation and SAMEER_GUIDE.md", "all"),
    ]))

    # ---- environment
    add("\n## 2. Environment (LIVE)\n")
    add(md_table(["Item", "Value"], list(env_info().items())))

    # ---- timeline
    add("\n## 3. What happened, in order\n")
    add(md_table(["Date", "What", "Result"], [
        ("22 to 26 Oct 2026", "Sprint 1 started late; work split tracker and proposal read", "Plan: M1 to M5 for the models track"),
        ("27 Oct 2026", "M1 to M4 code written and tested (17 tests); tools set up in Google Colab with the project on Drive", "All tests passed"),
        ("27 Oct 2026", "Small timing check (20k and 60k rows) on Colab to prove the script runs", "Worked; numbers not used"),
        ("28 Oct 2026", "Real timing test on 1,000,000 and 4,000,000 synthetic rows, target 14,000,000", "See section 7"),
        ("28 Oct 2026", "RF settings chosen and written into config.yaml, frozen set to true", "See section 6"),
        ("28 Oct 2026", "Tests re-run and runner dry run executed twice (second time all runs skipped = resume works)", "17 passed; resume confirmed"),
        ("Pending", "M5 first real run", "Blocked: frames.parquet, split files, supervisor decision S3 (tolerance)"),
    ]))

    # ---- M1
    add("\n## 4. Task by task\n")
    add("### M1 - Metrics (`src/metrics.py`, `tests/test_metrics.py`)\n")
    add("**What:** the evaluation numbers of the proposal, written from the formulas and not imported from a library.\n")
    add("**Formulas used** (rows of the confusion matrix = true class, columns = predicted class):\n")
    add("- precision_c = TP / (TP + FP); recall_c = TP / (TP + FN); F1_c = 2TP / (2TP + FP + FN)\n"
        "- macro-F1 = (1/C) x sum of F1_c over all C = 5 classes; accuracy = correct / all\n"
        "- class weight w_c = N / (C x N_c) on the training side (equals scikit-learn `class_weight='balanced'`)\n"
        "- all-Normal baseline: macro-F1 = 2 pi / (C (1 + pi)), accuracy = pi, where pi is the Normal share of the test set\n")
    add("**Decisions:** a class with no true frames has recall shown as blank (NaN), not 0; a class never predicted has "
        "precision blank; F1 of an undefined class counts as 0, so macro-F1 always averages over all 5 classes "
        "(same as scikit-learn with zero_division=0).\n")
    add("**Check against the tracker:** the tests confirm that pi = 0.9944 and C = 5 give 0.1994. Note that 0.9944 is the "
        "CICIoV2024 share from the base paper; the HCRL Normal share will be different, so every real run prints its own baseline.\n")
    add("**Functions (LIVE, read from the source):**\n")
    add(md_table(["Function", "What it does"], functions_of(ROOT / "src" / "metrics.py")))

    add("\n### M2 - Models and config (`src/models.py`, `config.yaml`, `tests/test_models.py`)\n")
    add("**What:** the three models, built from one place with settings that come only from `config.yaml`.\n")
    add("- **LR:** StandardScaler + LogisticRegression, inside one pipeline, so the scaler is fitted on the training rows only.\n"
        "- **DT:** DecisionTreeClassifier on the raw integer columns, no depth limit.\n"
        "- **RF:** RandomForestClassifier on the raw integer columns, bootstrap on, `max_samples` set from the M3 test.\n"
        "- All three: `class_weight='balanced'`, `random_state` = the run's seed (42 to 46). Same settings for P1 and P2.\n")
    add("**Safety flags in config.yaml:** `frozen` (RF settings written down) and `validity.tolerance_fixed` "
        "(supervisor decision at S3). The runner refuses to touch real data until both are true.\n")
    add("**Function (LIVE):**\n")
    add(md_table(["Function", "What it does"], functions_of(ROOT / "src" / "models.py")))

    add("\n### M3 - Timing and memory test (`src/timing_test.py`)\n")
    add("**What:** shows whether LR, DT and RF can be trained on about 14 million rows with the RAM and time available, "
        "without looking at any real label.\n")
    add("**How it works:** it generates synthetic CAN-like data (10 columns, 5 classes), fits each model in its own subprocess "
        "at each size (so memory numbers are clean), records fit time, prediction time, peak RAM and tree size, extrapolates to the "
        "target size (time: log-log fit; RAM: linear fit), and tests Random Forest candidates (n_estimators, max_samples) against a "
        "time budget (60 min per fit) and a RAM budget. Results are written to new files and never overwrite old ones.\n")
    add("**Command used:** `python -m src.timing_test --sizes 1000000 4000000 --target 14000000 --ram-gb 9 --cores 2 --budget-min 60`\n")
    add("**Limits (say these in the report):** synthetic data only has the right shape, so real tree size and time may differ; "
        "estimates are rough and optimistic. The first real RF run gives the true time.\n")

    add("\n### M4 - Experiment runner (`src/run_experiments.py`, `tests/test_runner.py`)\n")
    add("**What:** runs every (model, protocol, seed) combination and appends one row per run to `results/runs.csv`.\n")
    add("**Input assumptions (to be confirmed with the teammates):** `data/frames.parquet` with columns id, dlc, d0 to d7, label "
        "(names Normal/DoS/Fuzzy/RPM/Gear or 0 to 4); split files `splits/P1_seed42.npy` and `splits/P2_seed42.npy` (and 43 to 46) "
        "holding the TEST row positions; training rows are all other rows.\n")
    add("**Behaviour:**\n"
        "- One row per run with date, model, protocol, seed, status, validity, accuracy, macro-F1, per-class precision, recall and F1, "
        "n_train, n_test, Normal share of the test set, the all-Normal baseline, fit and predict seconds, missing classes, config hash, note.\n"
        "- A safety-net validity check per split: every class present in train and test and inside the test-share tolerance; an invalid split "
        "is recorded as `skipped_invalid` and not trained (proposal: invalid splits are not forced into the comparison).\n"
        "- Resume: finished runs are skipped when the same command is started again; a crashed run is stored as `error` and retried.\n"
        "- The results file is only appended to, never rewritten; one confusion matrix per run is saved in `results/confusion/`.\n"
        "- Guards: a real run is refused until `frozen` and `tolerance_fixed` are true; dry runs cannot write to `runs.csv`.\n"
        "- Convergence warnings (for example from LR) are written into the `note` column.\n")
    add("**Functions (LIVE):**\n")
    add(md_table(["Function", "What it does"], functions_of(ROOT / "src" / "run_experiments.py")))

    # ---- config
    add("\n## 5. Fixed configuration (LIVE, `config.yaml`)\n")
    cfg_text = read_text(ROOT / "config.yaml")
    add("```yaml\n" + cfg_text.strip() + "\n```\n")
    if cfg_text:
        add(f"SHA-256 of config.yaml (first 12 characters): `{hashlib.sha256(cfg_text.encode()).hexdigest()[:12]}` "
            "(the runner stores its own hash of the model section and seeds in every result row).\n")

    add("## 6. Decision on the Random Forest settings\n")
    add("The timing test recommended the first candidate inside the budgets. The team member chose one row lower: "
        "**n_estimators = 100, max_samples = 0.5**, because the estimate for full-size samples is above half of the 60-minute budget, "
        "estimates from synthetic data are optimistic, and 10 RF runs are needed (2 protocols x 5 seeds). "
        "The settings are identical for P1 and P2 and were fixed before any real result existed. "
        "The report must state that each tree sees half of the training rows.\n")

    # ---- timing
    add("## 7. Timing test results (LIVE)\n")
    rec = latest("results/timing_recommendation*.md")
    tab = latest("results/timing_table*.csv")
    if rec:
        add(f"File: `{rec.relative_to(ROOT)}` (last modified {datetime.fromtimestamp(rec.stat().st_mtime):%Y-%m-%d %H:%M})\n")
        # demote the headings of the embedded file so section numbering stays intact
        embedded = re.sub(r"^## ", "##### ", read_text(rec).strip(), flags=re.M)
        embedded = re.sub(r"^# ", "#### ", embedded, flags=re.M)
        add(embedded + "\n")
    else:
        add("_No `results/timing_recommendation*.md` found. Run the real timing test first._\n")
    if tab:
        with open(tab, newline="") as f:
            rows = list(csv.DictReader(f))
        add(f"\nRaw measurements: `{tab.relative_to(ROOT)}`\n")
        keep = ["model", "n_rows", "n_estimators", "fit_s", "predict_s", "rss_peak_fit_mb", "nodes_per_tree"]
        add(md_table(keep, [[r.get(k, "") for k in keep] for r in rows]))
    others = [p for p in glob.glob(str(ROOT / "results" / "timing_*")) if Path(p) not in (rec, tab)]
    if others:
        add("\nOther timing files present (earlier runs are kept, never overwritten): " + ", ".join(f"`{Path(p).name}`" for p in sorted(others)))

    # ---- tests
    add("\n\n## 8. Test results (LIVE, executed now)\n")
    code, out = sh([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"])
    per_file, summary = parse_pytest(out)
    if per_file:
        add(f"Summary: **{summary}**\n")
        for fname, tests in per_file.items():
            add(f"\n`{fname}`\n")
            add(md_table(["Test", "Result"], tests))
    else:
        add("Tests could not be executed here. Output:\n\n```\n" + out[-1500:] + "\n```")

    # ---- dry run
    add("\n\n## 9. Runner dry run (LIVE, synthetic data only, executed now)\n")
    add("A small dry run on 20,000 synthetic rows. **These numbers mean nothing for the project**, they only prove that the code works, "
        "writes the expected columns and resumes correctly.\n")
    tmp = Path(tempfile.mkdtemp(prefix="doc_dryrun_"))
    outcsv = tmp / "dry.csv"
    cmd = [sys.executable, "-m", "src.run_experiments", "--synthetic", "20000", "--models", "LR", "DT", "RF",
           "--seeds", "42", "--out", str(outcsv)]
    c1, o1 = sh(cmd)
    c2, o2 = sh(cmd)
    if outcsv.exists():
        with open(outcsv, newline="") as f:
            rows = list(csv.DictReader(f))
        keep = ["model", "protocol", "seed", "status", "valid", "accuracy", "macro_f1", "baseline_macro_f1", "fit_seconds", "n_train", "n_test", "note"]
        def fmt(v):
            try:
                return f"{float(v):.4f}" if "." in v else v
            except ValueError:
                return v
        add(md_table(keep, [[fmt(r.get(k, "")) for k in keep] for r in rows]))
        add(f"\nColumns in the results file ({len(rows[0])}): " + ", ".join(f"`{k}`" for k in rows[0].keys()))
        add("\n\nSecond start of the same command (resume check):\n\n```\n" + "\n".join(o2.strip().splitlines()[-4:]) + "\n```")
    else:
        add("Dry run did not produce a file. Output:\n\n```\n" + o1[-1500:] + "\n```")
    real_guard_code, real_guard = sh([sys.executable, "-m", "src.run_experiments", "--frames", "does_not_exist.parquet"])
    add("\nGuard check: starting a REAL run with the current config gives exit code "
        f"`{real_guard_code}`:\n\n```\n{real_guard.strip()[-500:]}\n```\n")

    # ---- rules and limits
    add("\n## 10. Rules followed and known limits\n")
    add(md_table(["Rule or limit", "How it is handled"], [
        ("No real data before the tolerance is fixed (ground rule 3)", "Runner refuses real runs until `tolerance_fixed` is true; all M1 to M4 work used synthetic data only"),
        ("Hyperparameters fixed before results (ground rule 4)", "`frozen: true` written after the timing test and before any real result; same values for P1 and P2"),
        ("Results files are never overwritten (ground rule 6)", "Timing files get a timestamp if a file exists; runner only appends; confusion matrices are not overwritten"),
        ("Check results in order (ground rule 8)", "Procedure in docs/SAMEER_GUIDE.md, Step 9"),
        ("Timing estimates", "From synthetic data, rough and optimistic; re-check with the real number of signatures and on the first real RF run"),
        ("Input formats", "Assumed (see M4); to be confirmed with the teammates who produce frames.parquet and the split files"),
        ("LR convergence on 14M real rows", "Not tested on real data; a ConvergenceWarning is recorded in the `note` column and must be reported, not hidden by changing max_iter afterwards"),
        ("Dummy splits in src/synthetic.py", "Only for dry runs; the official splits come from the splits track"),
    ]))

    # ---- status
    add("\n## 11. Status and sign-off\n")
    add(md_table(["Task", "Built", "Tested", "Teammate check (name, date)", "Status"], [
        ("M1 metrics", "yes", "yes (see section 8)", "", "Done once checked"),
        ("M2 models and config", "yes", "yes (see section 8)", "", "Done once checked"),
        ("M3 timing test", "yes", "yes, real run done 28 Oct 2026", "", "Done once checked"),
        ("M4 runner", "yes", "yes, dry run and resume (section 9)", "", "Done once checked"),
        ("M5 first real run", "-", "-", "-", "Blocked: frames.parquet, splits, S3 tolerance"),
    ]))
    add("\nWhat is needed next: `data/frames.parquet`, the split files, the agreed tolerance (set `tolerance_fixed: true` and the min and max test shares in `config.yaml`), then:\n\n"
        "```\npython -m src.run_experiments --seeds 42 --models LR DT\n```\n")
    add("## 12. How to reproduce everything\n")
    add("```\n%cd /content/drive/MyDrive/FYDP/files/fydp_models_sameer/fydp_models   # Colab\n"
        "pip install -r requirements.txt\npython -m pytest -v\n"
        "python -m src.timing_test --sizes 1000000 4000000 --target 14000000 --ram-gb 9 --cores 2 --budget-min 60\n"
        "python -m src.run_experiments --synthetic 200000 --models LR DT RF --seeds 42\n"
        "python -m src.make_documentation\n```\n")
    return "\n".join(L)


def to_html(md_text: str, title: str) -> str:
    try:
        import markdown
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except Exception:
        body = "<pre>" + html.escape(md_text) + "</pre>"
    css = ("body{font-family:Arial,Helvetica,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;line-height:1.5;color:#111}"
           "table{border-collapse:collapse;margin:10px 0;font-size:13px}th,td{border:1px solid #bbb;padding:4px 8px;text-align:left;vertical-align:top}"
           "th{background:#eee}code,pre{background:#f5f5f5;font-size:12px}pre{padding:8px;overflow-x:auto}h1,h2,h3{margin-top:1.4em}")
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{body}</body></html>"


def main():
    text = build()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / f"M1_M4_documentation_{stamp}.md"
    html_path = out_dir / f"M1_M4_documentation_{stamp}.html"
    md_path.write_text(text, encoding="utf-8")
    html_path.write_text(to_html(text, "FYDP M1-M4 documentation"), encoding="utf-8")
    print("Documentation written:")
    print("  ", md_path)
    print("  ", html_path)
    print("On Drive: MyDrive/" + str(ROOT).split("/MyDrive/")[-1] + "/docs" if "/MyDrive/" in str(ROOT) else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
