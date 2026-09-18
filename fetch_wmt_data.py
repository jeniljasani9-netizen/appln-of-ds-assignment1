import csv
from collections import Counter
from datasets import load_dataset

YEAR = 2018
PER_LP = 300
TARGET_LPS = ["cs-en", "de-en", "et-en", "fi-en", "ru-en", "tr-en", "zh-en"]
OUT = "data/wmt18_daseg.csv"

print(f"streaming dataset, looking for year={YEAR}, to-English pairs...")

ds = load_dataset(
    "RicardoRei/wmt-da-human-evaluation",
    split="train",
    streaming=True,
)

kept = {lp: [] for lp in TARGET_LPS}
scanned = 0

for row in ds:
    scanned += 1
    if scanned % 200000 == 0:
        have = sum(len(v) for v in kept.values())
        print(f"  scanned {scanned:,} rows, kept {have}")

    if row.get("year") != YEAR:
        continue
    lp = row.get("lp")
    if lp not in kept or len(kept[lp]) >= PER_LP:
        continue
    if not row.get("mt") or not row.get("ref"):
        continue

    kept[lp].append(row)

    if all(len(v) >= PER_LP for v in kept.values()):
        print("  got enough for every language pair, stopping early")
        break

rows = [r for v in kept.values() for r in v]
print(f"\nscanned {scanned:,} rows total, keeping {len(rows)}")
print("per language pair:", Counter(r["lp"] for r in rows))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["lp", "src", "mt", "ref", "raw", "score"])
    for r in rows:
        w.writerow([r["lp"], r["src"], r["mt"], r["ref"], r["raw"], r["score"]])

print(f"wrote {OUT}")

if rows:
    ex = rows[0]
    print("\nexample row:")
    print("  lp   :", ex["lp"])
    print("  mt   :", ex["mt"][:90])
    print("  ref  :", ex["ref"][:90])
    print("  raw  :", ex["raw"], "(human DA score)")
    print("  score:", ex["score"], "(z-normalised)")
