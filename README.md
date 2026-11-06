# Rules vs. LLMs: Income Tax & Data Quality Engines

Take-home technical assessment for **Fire AI** evaluating the boundaries between deterministic rules, statistical algorithms, and Large Language Models.

---

## 1. Executive Summary

This repository contains two production-grade projects demonstrating when a deterministic rule is the right tool, when an LLM earns its cost, and when the answer is a hybrid:

- **Project A — Income-Tax Engine (Rules vs. LLMs):**
  A statutory income tax calculator for salaried individuals in India under FY 2024-25 / AY 2025-26 (and FY 2025-26 Budget updates). Evaluates three implementations:
  1. `hand_coded.py`: Deterministic statutory rules engine (**13 µs latency, $0.00 cost, 100% accuracy**).
  2. `llm_query_time.py`: Live Gemini query-time engine reading raw legal clauses (**2.26s latency, ~$0.50/1k requests**).
  3. `compiled_rules_v1.py`: LLM-compiled executable rules with human-in-the-loop verification (**13 µs latency, $0.00 inference cost, 100% accuracy**).
  Includes a natural language / Hinglish conversational parser (`nl_parser.py`) that strictly generates clarifying questions when critical tax information is ambiguous or missing rather than hallucinating.

- **Project B — Data-Quality Pipeline (Rules vs. Statistics vs. LLMs):**
  A 3-tier data-quality firewall designed for 50M rows/day across 120 tenant schemas evaluated on 505,000 real-world NYC 311 service request records. Evaluates 36 detector combinations across 12 corruption types:
  - **Tier 1 (Rules):** Evaluates 100% of rows for structural, schema, coordinate bounding-box, temporal, and postal anomalies at **>200,000 rows/sec** and **$0.00 cost**.
  - **Tier 2 (Statistical):** Sliding-window Levenshtein similarity for near-duplicates and per-agency Z-scores for null-rate drift at **$0.00 cost**.
  - **Tier 3 (Targeted LLM Review):** Gemini Flash review strictly gated to the ~1.5% of rows with semantic ambiguity (resolution text vs. status contradictions, unlisted complaint descriptors, and unformatted person-name PII).
  **Result:** **98.8% catch rate**, **<0.15% false-alarm rate**, and **$0.18 per 1M rows** (**99.4% cost savings** vs. pure LLM).

---

## 2. Repository Layout

```
rules-vs-llms/
├── README.md
├── requirements.txt
├── .env.example                  # GEMINI_API_KEY=your_key_here
├── Makefile                      # Reproduces tests and generates comparison diffs
├── RESULTS_A.md                  # Authoritative comparison table & Budget 2025 change log
├── RESULTS_B.md                  # Authoritative benchmark (36 cells) & raw data profiling
├── DECISION_MATRIX.md            # Component-by-component rule vs LLM decision matrix
├── AI_USAGE.md                   # AI usage log with 3 failure case studies
│
├── project_a/
│   ├── data/
│   │   ├── rules_fy2024_25.md    # Statutory rules extracted clause-by-clause
│   │   └── rules_fy2025_26.md    # Budget 2025 / FY 2025-26 statutory updates
│   ├── implementations/
│   │   ├── hand_coded.py         # (a) Hand-coded deterministic rules engine
│   │   ├── llm_query_time.py     # (b) LLM reads rules + computes live per request
│   │   └── llm_compiled.py       # (c) LLM compiles rules once -> compiled_rules_v1.py
│   ├── compiled_rules/
│   │   └── compiled_rules_v1.py  # Human-reviewed compiled rules (100% match)
│   ├── parser/
│   │   └── nl_parser.py          # Conversational parser + anti-hallucination clarifying Qs
│   ├── common/
│   │   ├── schema.py             # Dataclasses: TaxpayerInput, TaxResult
│   │   └── clause_refs.py        # Statutory clause citations for legal auditability
│   ├── tests/
│   │   ├── test_cases.json       # 210 test cases (>=30 verified against official calc)
│   │   ├── nl_descriptions.json  # 50 plain-language user descriptions (English & Hinglish)
│   │   └── test_accuracy.py      # Pytest suite running all 210 test cases
│   └── measure.py                # Harness: accuracy, determinism, latency, cost, auditability
│
├── project_b/
│   ├── data/
│   │   └── nyc_311_2023.csv      # (Gitignored ~314MB dataset; fetch script provided)
│   ├── profiling/
│   │   └── profile_raw_data.py   # Task 1: Scans 505k raw records for natural defects
│   ├── injector/
│   │   └── corruption_injector.py# Controlled injection of all 12 problem types
│   ├── detectors/
│   │   ├── rule_based.py         # Tier 1: Declarative rules & schema checks
│   │   ├── statistical.py        # Tier 2: Distribution drift & sliding-window fuzzy matching
│   │   └── llm_review.py         # Tier 3: Gemini Flash semantic reviewer with failover
│   ├── pipeline/
│   │   └── hybrid_pipeline.py    # Task 5: Cascading 3-tier hybrid pipeline
│   ├── tests/
│   │   └── test_detectors.py     # Pytest suite covering all 12 detectors (19 tests)
│   └── measure.py                # Harness: 36 detector-tier combinations benchmark
│
├── scripts/
│   ├── fetch_nyc_311_data.py     # Automated fetch script for NYC Open Data 2023 extract
│   └── run_all.sh                # Single command to reproduce tests and benchmarks
│
└── docs/
    ├── final_design_a.md         # Task 6 (Project A): 100k user architecture doc (<= 2 pages)
    └── final_design_b.md         # Task 6 (Project B): 50M rows/day architecture doc (<= 2 pages)
```

---

## 3. Environment Setup

### 3.1 Python Environment
Requires **Python 3.10+** (tested on Python 3.11):
```bash
# Clone the repository
git clone <repo-url>
cd rules-vs-llms

# Install required dependencies
pip install -r requirements.txt
```

### 3.2 Configure API Keys
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## 4. Dataset Setup (Project B)

Project B evaluates real-world data quality on NYC 311 Service Requests created in calendar year 2023 (~314 MB, 505,000 rows).

To fetch the dataset automatically:
```bash
python scripts/fetch_nyc_311_data.py --out project_b/data/nyc_311_2023.csv
```
*Dataset Provenance:* NYC Open Data Socrata API, dataset ID `erm2-nwe9` (*311 Service Requests from 2010 to Present*), filtered to `created_date` between `2023-01-01` and `2023-12-31`.

> **Note on Fallback Generator:** If running without internet access or without the full CSV extract, `project_b/measure.py` includes a synthetic generator (`_generate_synthetic_clean_pool`) that creates clean baseline records following authentic NYC 311 distributions, schemas, coordinate bounding boxes, and ZIP codes.

---

## 5. How to Run Tests and Reproduce Results

### 5.1 Run Test Suites
All unit and regression tests run offline in seconds:
```bash
# Run Project A tax engine tests (210 test cases verified across hand_coded & compiled_rules)
python -m pytest project_a/tests/ -v

# Run Project B data quality detector tests (19 tests covering injector, rules, stats, and mock LLM)
python -m pytest project_b/tests/ -v

# Or run both simultaneously
make test
```

### 5.2 Reproduce Benchmarks
To regenerate all benchmark measurement tables across both projects:
```bash
make all
```
Or execute the automated script:
```bash
bash scripts/run_all.sh
```

### 5.3 Authoritative vs. Generated Results Note
- `RESULTS_A.md` and `RESULTS_B.md` are the **committed, authoritative submission documents** containing full empirical tables, task writeups, profiling defect discoveries, Budget 2025 change logs, and qualitative analyses.
- Running `make all` executes `project_a/measure.py` and `project_b/measure.py` and pipes raw output to `RESULTS_A_generated.md` and `RESULTS_B_generated.md`.
- To verify reproducibility without overwriting narrative documentation, run:
  ```bash
  make diff_results
  ```
  This diffs the newly generated numbers directly against the committed authoritative results.

---

## 6. Interview Quick Reference & Key Numbers

| Area | Deterministic Rules | Statistical / ML-Lite | LLM (Gemini Flash) | Hybrid Pipeline |
| :--- | :---: | :---: | :---: | :---: |
| **Project A: Tax Math** | **13 µs, $0.00, 100% exact** | N/A | 2.26s, $0.50/1k, risk of drift | **LLM Parser $\to$ Rule Engine** |
| **Project A: Parser** | Fails on informal language | N/A | **Translates Hinglish & flags missing info** | **Guarded LLM (refuses to guess)** |
| **Project B: Schema/Physics** | **>200k rows/s, $0, 100% recall** | High false alarms on schemas | 2–11 rows/s, $15–$56/1M, misses | **Pure Rules (Tier 1)** |
| **Project B: Distributions** | 0% recall (blind to aggregate drift) | **100% recall (Z-score & Levenshtein)** | $1.75–$8/1M, low throughput | **Pure Stats (Tier 2)** |
| **Project B: Free-Text Semantics** | 32–62% recall (dictionary/regex only) | 0–72% recall (misses subtle text bugs)| **100% recall (understands nuance)** | **Gated LLM on ~1.5% rows (Tier 3)** |
| **Project B: End-to-End** | 84.1% catch rate | Partial coverage | 78.2% catch rate, $32.00+/1M rows | **98.8% catch rate, $0.18/1M rows** |
