"""
Step 2: build the perturbation corpus.

For each source sentence we generate variants damaged in known ways.
Severity 0 is an exact copy; severity 5 reverses the meaning. A metric
that tracks meaning should score these in roughly descending order, so
any inversion is evidence of a specific weakness.

No human annotation is needed: the ordering comes from how each variant
was constructed.
"""

import csv, random, re
import nltk
from nltk.corpus import wordnet as wn

random.seed(42)

for pkg in ["wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

IN = "data/source_sentences.csv"
OUT = "data/perturbation_corpus.csv"

AUXILIARIES = {"is","are","was","were","can","could","will","would",
               "has","have","had","does","do","did","should","may","might"}

STOP = {"the","a","an","and","or","but","of","to","in","on","for","with",
        "as","by","at","from","that","this","these","those","it","we","our"}


def wordnet_pos(tag):
    if tag.startswith("JJ"): return wn.ADJ
    if tag.startswith("VB"): return wn.VERB
    if tag.startswith("NN"): return wn.NOUN
    if tag.startswith("RB"): return wn.ADV
    return None


def synonym_swap(words, tags):
    """Severity 1: replace a content word with a WordNet synonym."""
    idxs = list(range(len(words)))
    random.shuffle(idxs)
    for i in idxs:
        w, t = words[i], tags[i]
        if w.lower() in STOP or not w.isalpha():
            continue
        pos = wordnet_pos(t)
        if pos is None:
            continue
        for syn in wn.synsets(w.lower(), pos=pos):
            for lemma in syn.lemmas():
                cand = lemma.name().replace("_", " ")
                if cand.lower() != w.lower() and " " not in cand:
                    out = words[:]
                    out[i] = cand
                    return " ".join(out), f"{w} -> {cand}"
    return None, None


def scramble(words, tags):
    """Severity 2: same words, shuffled order (PAWS-style)."""
    if len(words) < 6:
        return None, None
    body = words[:-1] if words[-1] in ".!?" or words[-1].endswith(".") else words[:]
    out = body[:]
    for _ in range(20):
        random.shuffle(out)
        if out != body:
            break
    return " ".join(out), "word order shuffled"


def number_swap(words, tags):
    """Severity 3: change a specific fact (a number)."""
    for i, w in enumerate(words):
        m = re.fullmatch(r"(\d+)([.,]\d+)?%?", w)
        if m:
            old = int(m.group(1))
            new = old + random.choice([3, 7, 11, 25, 100])
            out = words[:]
            out[i] = w.replace(m.group(1), str(new), 1)
            return " ".join(out), f"{w} -> {out[i]}"
    return None, None


def antonym_swap(words, tags):
    """Severity 4: invert a descriptive word."""
    idxs = list(range(len(words)))
    random.shuffle(idxs)
    for i in idxs:
        w, t = words[i], tags[i]
        if not w.isalpha() or w.lower() in STOP:
            continue
        pos = wordnet_pos(t)
        if pos not in (wn.ADJ, wn.ADV, wn.VERB):
            continue
        for syn in wn.synsets(w.lower(), pos=pos):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    cand = ant.name().replace("_", " ")
                    if " " not in cand:
                        out = words[:]
                        out[i] = cand
                        return " ".join(out), f"{w} -> {cand}"
    return None, None


def negate(words, tags):
    """Severity 5: reverse the claim outright."""
    for i, w in enumerate(words):
        if w.lower() in AUXILIARIES:
            out = words[:i+1] + ["not"] + words[i+1:]
            return " ".join(out), f"inserted 'not' after '{w}'"
    body = " ".join(words)
    return "It is not the case that " + body[0].lower() + body[1:], "prepended denial"


TRANSFORMS = [
    ("identical",    0, lambda w, t: (" ".join(w), "unchanged")),
    ("synonym",      1, synonym_swap),
    ("scramble",     2, scramble),
    ("number_swap",  3, number_swap),
    ("antonym",      4, antonym_swap),
    ("negation",     5, negate),
]

rows, failures = [], {}

with open(IN, encoding="utf-8") as f:
    sources = list(csv.DictReader(f))

for sid, src in enumerate(sources):
    sent = src["sentence"]
    words = sent.split()
    tags = [t for _, t in nltk.pos_tag(words)]

    for name, sev, fn in TRANSFORMS:
        try:
            text, note = fn(words, tags)
        except Exception:
            text, note = None, None
        if text is None:
            failures[name] = failures.get(name, 0) + 1
            continue
        rows.append({
            "sentence_id": sid,
            "category": src["category"],
            "reference": sent,
            "perturbation": name,
            "severity": sev,
            "candidate": text,
            "note": note,
        })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["sentence_id","category","reference",
                                      "perturbation","severity","candidate","note"])
    w.writeheader()
    w.writerows(rows)

print(f"source sentences : {len(sources)}")
print(f"variants produced: {len(rows)}")
print(f"wrote {OUT}\n")

print("per transformation:")
for name, sev, _ in TRANSFORMS:
    n = sum(1 for r in rows if r["perturbation"] == name)
    miss = failures.get(name, 0)
    print(f"  severity {sev}  {name:<12} {n:>4} made, {miss:>3} not applicable")

print("\nexample set (sentence 0):")
for r in rows:
    if r["sentence_id"] == 0:
        print(f"  [{r['severity']}] {r['perturbation']:<12} {r['candidate'][:78]}")
