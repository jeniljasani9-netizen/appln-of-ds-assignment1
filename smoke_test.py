import torch, transformers, bert_score
from bert_score import BERTScorer

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("bert_score:", bert_score.__version__)

reference = "the weather is cold today"
candidates = [
    "the weather is cold today",
    "it is freezing today",
    "the stock market closed higher",
]

scorer = BERTScorer(lang="en", rescale_with_baseline=True)
P, R, F1 = scorer.score(candidates, [reference] * 3)

print("\nconfig:", scorer.hash)
print("reference:", reference)
for cand, p, r, f in zip(candidates, P, R, F1):
    print(f"P={p:.4f} R={r:.4f} F1={f:.4f}  <- {cand}")