"""
Step 3: score the perturbation corpus with BERTScore and BLEU.

Each candidate is scored against its own source sentence as reference.
We then check whether scores fall as severity rises, and how often each
metric ranks a damaged sentence above a less damaged one.
"""

import csv
from collections import defaultdict
from statistics import mean, stdev

from scipy.stats import spearmanr
from sacrebleu.metrics import BLEU
from bert_score import BERTScorer

IN = "data/perturbation_corpus.csv"
OUT = "results_perturbation.csv"

with open(IN, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"scoring {len(rows)} candidate-reference pairs\n")

scorer = BERTScorer(lang="en", rescale_with_baseline=True, batch_size=16)
bleu = BLEU(effective_order=True)
print("config:", scorer.hash, "\n")

cands = [r["candidate"] for r in rows]
refs = [r["reference"] for r in rows]

_, _, F1 = scorer.score(cands, refs, verbose=True)
for r, f in zip(rows, F1):
    r["bertscore"] = f.item()
    r["bleu"] = bleu.sentence_score(r["candidate"], [r["reference"]]).score

by_pert = defaultdict(list)
for r in rows:
    by_pert[(int(r["severity"]), r["perturbation"])].append(r)

hdr = f"{'sev':>3} {'perturbation':<13} {'n':>4} {'BERTScore':>20} {'BLEU':>18}"
print(hdr)
print("-" * len(hdr))

summary = []
for (sev, name) in sorted(by_pert):
    g = by_pert[(sev, name)]
    bs = [r["bertscore"] for r in g]
    bl = [r["bleu"] for r in g]
    bs_sd = stdev(bs) if len(bs) > 1 else 0.0
    bl_sd = stdev(bl) if len(bl) > 1 else 0.0
    print(f"{sev:>3} {name:<13} {len(g):>4}   {mean(bs):>7.4f} +/- {bs_sd:<6.4f}"
          f"  {mean(bl):>6.2f} +/- {bl_sd:<6.2f}")
    summary.append([sev, name, len(g), mean(bs), bs_sd, mean(bl), bl_sd])

sev_all = [int(r["severity"]) for r in rows]
bs_all = [r["bertscore"] for r in rows]
bl_all = [r["bleu"] for r in rows]

print("\nSpearman correlation with severity (want strongly negative):")
print(f"  BERTScore {spearmanr(sev_all, bs_all)[0]:+.4f}")
print(f"  BLEU      {spearmanr(sev_all, bl_all)[0]:+.4f}")

# The key diagnostic: does negation get scored like a faithful paraphrase?
syn = {r["sentence_id"]: r["bertscore"] for r in rows if r["perturbation"] == "synonym"}
neg = {r["sentence_id"]: r["bertscore"] for r in rows if r["perturbation"] == "negation"}
scr = {r["sentence_id"]: r["bertscore"] for r in rows if r["perturbation"] == "scramble"}

shared = set(syn) & set(neg)
neg_above_syn = sum(1 for i in shared if neg[i] >= syn[i])
gap = mean([syn[i] - neg[i] for i in shared])

shared2 = set(scr) & set(neg)
neg_above_scr = sum(1 for i in shared2 if neg[i] >= scr[i])

print(f"\nnegation scored at or above synonym swap: {neg_above_syn}/{len(shared)} "
      f"({100*neg_above_syn/len(shared):.1f}%)")
print(f"mean BERTScore gap, synonym minus negation: {gap:+.4f}")
print(f"negation scored at or above word scramble: {neg_above_scr}/{len(shared2)} "
      f"({100*neg_above_scr/len(shared2):.1f}%)")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["severity","perturbation","n","bertscore_mean","bertscore_sd",
                "bleu_mean","bleu_sd"])
    w.writerows(summary)

with open("data/perturbation_scored.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nwrote {OUT} and data/perturbation_scored.csv")
