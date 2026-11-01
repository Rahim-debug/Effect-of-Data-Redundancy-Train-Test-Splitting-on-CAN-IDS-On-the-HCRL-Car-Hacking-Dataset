import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FYDP_DATA", ROOT / "data"))   # override for tests
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SAMPLE = DATA / "sample"

SEEDS = [42, 43, 44, 45, 46]
LABELS = ["Normal", "DoS", "Fuzzy", "RPM", "Gear"]
FEATURES = ["id", "dlc"] + [f"d{i}" for i in range(8)]
