"""
Step 1 of new dataset construction: collect source sentences.

Pulls recent abstracts from the arXiv API and splits them into clean
single sentences. Scientific abstracts are a deliberate domain shift:
BERTScore was validated on news translation and image captions.
"""

import csv, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

API = "http://export.arxiv.org/api/query"
CATEGORIES = ["cs.CL", "q-bio.NC", "econ.EM"]   # deliberately varied fields
PER_CAT = 100                                   # abstracts per category
TARGET = 200                                    # sentences to keep
OUT = "data/source_sentences.csv"

ATOM = "{http://www.w3.org/2005/Atom}"


def fetch(category, n):
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": n,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    req = urllib.request.Request(f"{API}?{params}",
                                 headers={"User-Agent": "COMP8240-student-project"})
    with urllib.request.urlopen(req, timeout=60) as r:
        xml = r.read()
    root = ET.fromstring(xml)
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        node = entry.find(f"{ATOM}summary")
        if node is not None and node.text:
            out.append(" ".join(node.text.split()))
    return out


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)


def is_clean(s):
    words = s.split()
    if not (10 <= len(words) <= 30):
        return False
    if not s.endswith("."):
        return False
    if re.search(r"[\\${}^_|]|\d+\)|et al|http", s):   # LaTeX, citations, urls
        return False
    if s.count("(") != s.count(")"):
        return False
    return True


sentences, seen = [], set()

for cat in CATEGORIES:
    print(f"fetching {PER_CAT} abstracts from {cat}...")
    try:
        abstracts = fetch(cat, PER_CAT)
    except Exception as e:
        print(f"  failed: {e}")
        continue
    print(f"  got {len(abstracts)} abstracts")

    for a in abstracts:
        for s in split_sentences(a):
            s = s.strip()
            if is_clean(s) and s not in seen:
                seen.add(s)
                sentences.append({"category": cat, "sentence": s})
    print(f"  running total of clean sentences: {len(sentences)}")
    time.sleep(3)          # arXiv asks for 3s between API calls

sentences = sentences[:TARGET]

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["category", "sentence"])
    w.writeheader()
    w.writerows(sentences)

print(f"\nkept {len(sentences)} sentences, wrote {OUT}")
print("\nfirst three:")
for s in sentences[:3]:
    print(f"  [{s['category']}] {s['sentence']}")
