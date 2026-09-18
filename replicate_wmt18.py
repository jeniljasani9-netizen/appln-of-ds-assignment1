"""
Replication: does BERTScore correlate with human judgement better than BLEU?

For each WMT18 to-English language pair we score every machine translation
against its human reference twice, once with BERTScore and once with
sentence-BLEU, then measure how well each metric's scores line up with the
human Direct Assessment scores.

Pearson r  = strength of linear agreement
Kendall tau = agreement in ranking (what the paper reports at segment level)
"""

import csv
from collections import defaultdict

from scipy.stats import pearsonr, kendalltau
from sacrebleu.metrics import BLEU
from bert_score import BERTScorer

IN = "data/wmt18_daseg.csv"
OUT = "results_wmt18.csv"

rows = defaultdict(list)
with open(IN, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows[r["lp"]].append(r)

print(f"loaded {sum(len(v) for v in rows.values())} rows "
      f"across {len(rows)} language pairs\n")

scorer = BERTScorer(lang="en", rescale_with_baseline=True)
bleu = BLEU(effective_order=True)
print("config:", scorer.hash, "\n")

header = f"{'lang pair':<10} {'n':>5} {'BERTScore r':>12} {'BLEU r':>9} {'BERTScore tau':>14} {'BLEU tau':>9}"
print(header)
print("-" * len(header))

results = []
all_bs, all_bl, all_hu = [], [], []

for lp in sorted(rows):
    data = rows[lp]
    cands = [d["mt"] for d in data]
    refs = [d["ref"] for d in data]
    human = [float(d["score"]) for d in data]

    _, _, F1 = scorer.score(cands, refs)
    bs = [f.item() for f in F1]
    bl = [bleu.sentence_score(c, [r]).score for c, r in zip(cands, refs)]

    bs_r = pearsonr(bs, human)[0]
    bl_r = pearsonr(bl, human)[0]
    bs_t = kendalltau(bs, human)[0]
    bl_t = kendalltau(bl, human)[0]

    print(f"{lp:<10} {len(data):>5} {bs_r:>12.4f} {bl_r:>9.4f} {bs_t:>14.4f} {bl_t:>9.4f}")
    results.append([lp, len(data), bs_r, bl_r, bs_t, bl_t])

    all_bs += bs
    all_bl += bl
    all_hu += human

print("-" * len(header))
ov = ["ALL", len(all_hu),
      pearsonr(all_bs, all_hu)[0], pearsonr(all_bl, all_hu)[0],
      kendalltau(all_bs, all_hu)[0], kendalltau(all_bl, all_hu)[0]]
print(f"{ov[0]:<10} {ov[1]:>5} {ov[2]:>12.4f} {ov[3]:>9.4f} {ov[4]:>14.4f} {ov[5]:>9.4f}")
results.append(ov)

wins = sum(1 for r in results[:-1] if r[4] > r[5])
print(f"\nBERTScore beats BLEU on Kendall tau in {wins} of {len(results)-1} language pairs")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["lp", "n", "bertscore_pearson", "bleu_pearson",
                "bertscore_kendall", "bleu_kendall"])
    w.writerows(results)
print(f"wrote {OUT}")
