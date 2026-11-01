"""Write tiny fake raw files in the HCRL layout so the pipeline can be tested without the dataset.
Usage: FYDP_DATA=/tmp/fake python tests/make_fake_raw.py
"""
import os, random
from pathlib import Path

raw = Path(os.environ["FYDP_DATA"]) / "raw"
raw.mkdir(parents=True, exist_ok=True)
random.seed(0)
hexb = lambda n: " ".join(f"{random.randrange(256):02x}" for _ in range(n))

with open(raw / "normal_run_data.txt", "w") as f:
    for i in range(3000):
        dlc = random.choice([8, 8, 8, 2])
        idv = random.choice([0x316, 0x153, 0x43F])
        f.write(f"Timestamp: {1478198376 + i * 0.001:.6f}        ID: {idv:04x}    000    DLC: {dlc}    {hexb(dlc)}\n")

for name, idfun in [("DoS_dataset.csv", lambda: 0), ("Fuzzy_dataset.csv", lambda: random.randrange(0x800)),
                    ("RPM_dataset.csv", lambda: 0x43F), ("gear_dataset.csv", lambda: 0x43F)]:
    with open(raw / name, "w") as f:
        for i in range(3000):
            attack = i % 3 == 0
            dlc = 8 if attack or random.random() < 0.8 else 2
            idv = idfun() if attack else random.choice([0x316, 0x153])
            data = ",".join(f"{random.randrange(256):02x}" for _ in range(dlc))
            f.write(f"{1478198376 + i * 0.001:.6f},{idv:04x},{dlc},{data},{'T' if attack else 'R'}\n")
print("fake raw files written to", raw)
