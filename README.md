# Replicating BERTScore

COMP8240 Applications of Data Science, Macquarie University, Session 2 2026.
Jenil Jasani (48779547).

A replication of **BERTScore: Evaluating Text Generation with BERT**
(Zhang, Kishore, Wu, Weinberger & Artzi, ICLR 2020), followed by two
extensions onto data the original paper did not use.

Reference implementation: https://github.com/Tiiiger/bert_score

## What was found

| Experiment | Result |
|---|---|
| WMT18 replication | BERTScore beats BLEU in all 7 to-English pairs. Overall Kendall tau 0.369 vs 0.238 |
| vs. published figures | Mean advantage over BLEU is +0.131 in this run and +0.131 in the paper |
| SummEval extension | Tracks relevance (r = 0.358) far better than factual consistency (r = 0.150) |
| Perturbation corpus | An antonym swap outscores its synonym counterpart 47% of the time |
| LLM-as-judge | A 0.5B judge reduces that to 13%, but scores scrambled word salad at 0.93 |

The two metrics fail in complementary ways: BERTScore is surface-sensitive
and meaning-blind, the LLM judge is the reverse.

## Setup

    pip install -r requirements.txt

All experiments run on CPU. No GPU required.

## Pipeline

Run in this order. Each script writes its own output file, and the
captured output of every run is committed alongside it.

| # | Script | Writes | Time |
|---|---|---|---|
| 1 | `smoke_test.py` | `smoke_test_output.txt` | 1 min |
| 2 | `fetch_wmt_data.py` | `data/wmt18_daseg.csv` | 2 min |
| 3 | `replicate_wmt18.py` | `results_wmt18.csv` | 20 min |
| 4 | `score_summeval.py` | `results_summeval.csv` | 60 min |
| 5 | `scrape_arxiv.py` | `data/source_sentences.csv` | 1 min |
| 6 | `build_perturbations.py` | `data/perturbation_corpus.csv` | 1 min |
| 7 | `score_perturbations.py` | `data/perturbation_scored.csv`, `results_perturbation.csv` | 5 min |
| 8 | `analyse_families.py` | `analyse_families_output.txt` | instant |
| 9 | `llm_judge.py` | `results_llm_judge.csv`, `data/perturbation_judged.csv` | 8 min |

Or run everything:

    bash run_all.sh

Steps 2 to 4 cover the replication and the existing-dataset extension.
Steps 5 to 9 build and evaluate the newly constructed dataset.

## Data

| File | Rows | Source |
|---|---|---|
| `data/wmt18_daseg.csv` | 2,100 | WMT18 Direct Assessment judgements, 300 per to-English pair |
| `data/source_sentences.csv` | 200 | Sentences from arXiv abstracts (cs.CL, q-bio.NC, econ.EM) |
| `data/perturbation_corpus.csv` | 1,008 | Six controlled perturbations per source sentence |
| `data/perturbation_scored.csv` | 1,008 | The above, with BERTScore and BLEU |
| `data/perturbation_judged.csv` | 303 | A 60-sentence subset, with LLM judge scores |

SummEval is pulled at runtime from the HuggingFace Hub and not stored here.

## Environment notes

Developed in a GitHub Codespace: 2 CPU cores, 8 GB RAM, Python 3.14.

Three things a reader reproducing this should know.

The `bert_score` package was last updated in 2024 but installs and runs
unmodified against current PyTorch and Transformers with no version
pinning required.

For `llm_judge.py`, **float32 is faster than bfloat16 on CPU**, because
PyTorch has no optimised bf16 CPU kernels. Loading the judge in bfloat16
ran roughly 40 times slower.

Qwen2.5-1.5B does not fit in 8 GB in float32, so the judge uses
Qwen2.5-0.5B-Instruct. This is far smaller than the GPT-4 class models
used in published LLM-as-judge work, so the judge results are a floor
rather than a ceiling.

## Known limitations

Number-swap perturbations applied to only 24 of 200 sentences, since most
abstract sentences contain no digits. In the LLM judge subset this falls
to 8, too few to draw conclusions from.

WMT18 figures use 300 sentences per language pair where the paper used
thousands, and standard Kendall tau where the paper used daRR. The
direction of the finding replicates; the individual values are not
directly comparable.
