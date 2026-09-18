"""Separate surface-preserving meaning changes from surface disruption."""
import csv
from statistics import mean
from scipy.stats import spearmanr

rows = list(csv.DictReader(open("data/perturbation_scored.csv", encoding="utf-8")))
for r in rows:
    r["bertscore"] = float(r["bertscore"]); r["bleu"] = float(r["bleu"])
    r["severity"] = int(r["severity"])

MEANING = {"number_swap", "antonym", "negation"}   # words kept, meaning changed
SURFACE = {"scramble"}                             # words kept, order destroyed

base = mean(r["bertscore"] for r in rows if r["perturbation"] == "synonym")
print(f"baseline (synonym swap, meaning preserved): {base:.4f}\n")

print("cost of each edit, relative to a meaning-preserving synonym swap:")
for p in ["number_swap", "antonym", "negation", "scramble"]:
    g = [r for r in rows if r["perturbation"] == p]
    print(f"  {p:<12} n={len(g):>3}  BERTScore drop {base - mean(r['bertscore'] for r in g):+.4f}")

sub = [r for r in rows if r["perturbation"] in MEANING | {"identical", "synonym"}]
print(f"\nexcluding word scrambling ({len(sub)} pairs), Spearman with severity:")
print(f"  BERTScore {spearmanr([r['severity'] for r in sub], [r['bertscore'] for r in sub])[0]:+.4f}")
print(f"  BLEU      {spearmanr([r['severity'] for r in sub], [r['bleu'] for r in sub])[0]:+.4f}")

ant = {r["sentence_id"]: r["bertscore"] for r in rows if r["perturbation"] == "antonym"}
syn = {r["sentence_id"]: r["bertscore"] for r in rows if r["perturbation"] == "synonym"}
shared = set(ant) & set(syn)
worse = sum(1 for i in shared if ant[i] >= syn[i])
print(f"\nantonym scored at or ABOVE its synonym counterpart: "
      f"{worse}/{len(shared)} ({100*worse/len(shared):.1f}%)")
