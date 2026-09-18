"""
Extension 2: apply BERTScore to SummEval, a dataset the paper never used.

SummEval pairs 16 machine summaries of 100 CNN/DailyMail articles with
expert human ratings on four separate dimensions: coherence, consistency,
fluency and relevance. Because the dimensions are rated separately we can
ask which aspect of quality BERTScore actually tracks.

Hypothesis from the project proposal: BERTScore should correlate better
with relevance than with consistency, since a summary can closely resemble
a reference while stating a fact incorrectly.
"""

import csv
from statistics import mean

from datasets import load_dataset
from scipy.stats import pearsonr, kendalltau
from sacrebleu.metrics import BLEU
from bert_score import BERTScorer

N_ARTICLES = 30     # of 100; keeps CPU runtime reasonable
N_REFS = 3          # human reference summaries to score against
OUT = "results_summeval.csv"
DIMS = ["coherence", "consistency", "fluency", "relevance"]

ds = load_dataset("mteb/summeval", split="test")
print("columns:", ds.column_names)
print("articles available:", len(ds), "\n")

missing = [c for c in ["machine_summaries", "human_summaries"] + DIMS
           if c not in ds.column_names]
if missing:
    raise SystemExit(f"unexpected schema, missing: {missing}")

cands, refs, human = [], [], {d: [] for d in DIMS}

for row in list(ds)[:N_ARTICLES]:
    machine = row["machine_summaries"]
    references = row["human_summaries"][:N_REFS]
    if not references:
        continue
    for i, summary in enumerate(machine):
        cands.append(" ".join(summary.split()))
        refs.append([" ".join(r.split()) for r in references])
        for d in DIMS:
            human[d].append(row[d][i])

print(f"scoring {len(cands)} machine summaries "
      f"against {N_REFS} references each\n")

scorer = BERTScorer(lang="en", rescale_with_baseline=True, batch_size=8)
bleu = BLEU(effective_order=True)
print("config:", scorer.hash, "\n")

_, _, F1 = scorer.score(cands, refs, verbose=True)
bs = [f.item() for f in F1]
bl = [max(bleu.sentence_score(c, [r]).score for r in rs)
      for c, rs in zip(cands, refs)]

hdr = f"{'dimension':<13} {'BERTScore r':>12} {'BLEU r':>9} {'BERTScore tau':>14} {'BLEU tau':>9}"
print(hdr)
print("-" * len(hdr))

rows = []
for d in DIMS:
    h = human[d]
    r1, r2 = pearsonr(bs, h)[0], pearsonr(bl, h)[0]
    t1, t2 = kendalltau(bs, h)[0], kendalltau(bl, h)[0]
    print(f"{d:<13} {r1:>12.4f} {r2:>9.4f} {t1:>14.4f} {t2:>9.4f}")
    rows.append([d, len(h), r1, r2, t1, t2])

print()
rel = next(r for r in rows if r[0] == "relevance")
con = next(r for r in rows if r[0] == "consistency")
print(f"relevance minus consistency, BERTScore Pearson: {rel[2] - con[2]:+.4f}")
print("(positive supports the proposal's hypothesis)")

wins = sum(1 for r in rows if r[4] > r[5])
print(f"BERTScore beats BLEU on Kendall tau in {wins} of {len(rows)} dimensions")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dimension", "n", "bertscore_pearson", "bleu_pearson",
                "bertscore_kendall", "bleu_kendall"])
    w.writerows(rows)
print(f"\nwrote {OUT}")
