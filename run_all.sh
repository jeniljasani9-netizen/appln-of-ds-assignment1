#!/usr/bin/env bash
# Runs the full pipeline in order. Roughly 100 minutes on 2 CPU cores.
set -e

echo "=== 1/9 feasibility smoke test ==="
python smoke_test.py | tee smoke_test_output.txt

echo "=== 2/9 fetch WMT18 human judgements ==="
python fetch_wmt_data.py

echo "=== 3/9 replicate WMT18 correlations (~20 min) ==="
python replicate_wmt18.py 2>&1 | tee replicate_wmt18_output.txt

echo "=== 4/9 SummEval extension (~60 min) ==="
python score_summeval.py 2>&1 | tee score_summeval_output.txt

echo "=== 5/9 scrape arXiv abstracts ==="
python scrape_arxiv.py

echo "=== 6/9 build perturbation corpus ==="
python build_perturbations.py

echo "=== 7/9 score perturbations (~5 min) ==="
python score_perturbations.py 2>&1 | tee score_perturbations_output.txt

echo "=== 8/9 analyse perturbation families ==="
python analyse_families.py 2>&1 | tee analyse_families_output.txt

echo "=== 9/9 LLM-as-judge (~8 min) ==="
python llm_judge.py 2>&1 | tee llm_judge_output.txt

echo "=== done ==="
