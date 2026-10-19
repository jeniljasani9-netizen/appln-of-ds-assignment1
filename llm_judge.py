"""LLM-as-judge on the perturbation corpus.

Asks whether a small instruction-tuned model distinguishes meaning
inversion better than BERTScore does. We read the probability the model
assigns to "Yes" rather than sampling text, so scores are continuous and
deterministic.

Config forced by a 2-core 8GB container: float32 (PyTorch has no fast
bf16 CPU kernels) and Qwen2.5-0.5B-Instruct (largest that fits).
"""

import csv, random, time
from collections import defaultdict
from statistics import mean, stdev

import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
IN = "data/perturbation_scored.csv"
OUT = "results_llm_judge.csv"
N_SENTENCES = 60
BATCH = 8

torch.set_num_threads(2)
random.seed(42)

rows = list(csv.DictReader(open(IN, encoding="utf-8")))
for r in rows:
    r["bertscore"] = float(r["bertscore"])
    r["severity"] = int(r["severity"])

ids = sorted({int(r["sentence_id"]) for r in rows})
keep = set(random.sample(ids, min(N_SENTENCES, len(ids))))
rows = [r for r in rows if int(r["sentence_id"]) in keep]
print(f"judging {len(rows)} pairs from {len(keep)} sentences\n", flush=True)

print(f"loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
print("loaded\n", flush=True)

def ids_for(word):
    out = set()
    for form in (word, " " + word, word.lower(), " " + word.lower()):
        enc = tok.encode(form, add_special_tokens=False)
        if enc:
            out.add(enc[0])
    return sorted(out)

YES, NO = ids_for("Yes"), ids_for("No")
print(f"yes {YES}  no {NO}\n", flush=True)

PROMPT = ("Reference: {ref}\nCandidate: {cand}\n\n"
          "Does the candidate preserve the meaning of the reference? "
          "Answer Yes or No.")

def build(r):
    msgs = [{"role": "user",
             "content": PROMPT.format(ref=r["reference"], cand=r["candidate"])}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

texts = [build(r) for r in rows]
scores = []
t0 = time.time()

for start in range(0, len(texts), BATCH):
    enc = tok(texts[start:start + BATCH], return_tensors="pt",
              padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    yes, no = probs[:, YES].sum(dim=1), probs[:, NO].sum(dim=1)
    scores.extend((yes / (yes + no)).tolist())
    done = len(scores)
    rate = done / (time.time() - t0)
    print(f"  {done}/{len(texts)} ({rate:.1f}/s, {(len(texts)-done)/rate:.0f}s left)",
          flush=True)

for r, s in zip(rows, scores):
    r["judge"] = s
print(f"\ntotal {time.time()-t0:.0f}s")

by_pert = defaultdict(list)
for r in rows:
    by_pert[(r["severity"], r["perturbation"])].append(r)

hdr = f"{'sev':>3} {'perturbation':<13} {'n':>4} {'BERTScore':>11} {'LLM judge':>11}"
print("\n" + hdr)
print("-" * len(hdr))

summary = []
for key in sorted(by_pert):
    sev, name = key
    g = by_pert[key]
    bs, jd = [x["bertscore"] for x in g], [x["judge"] for x in g]
    print(f"{sev:>3} {name:<13} {len(g):>4} {mean(bs):>11.4f} {mean(jd):>11.4f}")
    summary.append([sev, name, len(g), mean(bs),
                    stdev(bs) if len(bs) > 1 else 0.0,
                    mean(jd), stdev(jd) if len(jd) > 1 else 0.0])

sev_all = [r["severity"] for r in rows]
print("\nSpearman with severity (more negative is better):")
print(f"  BERTScore {spearmanr(sev_all, [r['bertscore'] for r in rows])[0]:+.4f}")
print(f"  LLM judge {spearmanr(sev_all, [r['judge'] for r in rows])[0]:+.4f}")

syn = {r["sentence_id"]: r for r in rows if r["perturbation"] == "synonym"}
ant = {r["sentence_id"]: r for r in rows if r["perturbation"] == "antonym"}
shared = sorted(set(syn) & set(ant))
bs_fail = sum(1 for i in shared if ant[i]["bertscore"] >= syn[i]["bertscore"])
jd_fail = sum(1 for i in shared if ant[i]["judge"] >= syn[i]["judge"])

print(f"\nantonym at or above its synonym counterpart ({len(shared)} pairs):")
print(f"  BERTScore {bs_fail}/{len(shared)} ({100*bs_fail/len(shared):.1f}%)")
print(f"  LLM judge {jd_fail}/{len(shared)} ({100*jd_fail/len(shared):.1f}%)")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["severity", "perturbation", "n", "bertscore_mean",
                "bertscore_sd", "judge_mean", "judge_sd"])
    w.writerows(summary)

with open("data/perturbation_judged.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nwrote {OUT} and data/perturbation_judged.csv")
