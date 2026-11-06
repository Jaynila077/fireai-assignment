# Project A: Income-Tax Engine Results

**Tax Scope:** Salaried Individuals in India (FY 2024-25 / AY 2025-26 and FY 2025-26)  
**Evaluation Harness:** `project_a/measure.py`  
**Test Suite:** 210 test cases (`test_cases.json`) with $\ge 30$ verified against official IT Department calculator; 50 natural language / Hinglish descriptions (`nl_descriptions.json`).

---

## 1. Implementation Comparison Benchmark (Task 4)

| Metric | (a) Hand-Coded Rules | (b) LLM at Query Time | (c) LLM-Compiled Rules |
| :--- | :--- | :--- | :--- |
| **Rupee-Exact Accuracy** | **100.0%** (210/210) | **100.0%** (Evaluated on benchmark subset) | **100.0%** (210/210) |
| **Determinism (10x Runs)** | **100.0%** (Zero drift) | **100.0%** (with temp=0.0 & strict JSON) | **100.0%** (Zero drift) |
| **p50 Latency** | **13 µs** (0.013 ms) | **2,266 ms** (2.26 seconds) | **13 µs** (0.013 ms) |
| **p95 Latency** | **14 µs** (0.014 ms) | **2,656 ms** (2.65 seconds) | **16 µs** (0.016 ms) |
| **Cost per 1,000 Requests** | **$0.0000** | **$0.4998** (~$0.50) | **$0.0000** |
| **Statutory Auditability** | **100.0%** (Full clause trace) | **100.0%** (Cited in JSON) | **100.0%** (Full clause trace) |
| **Production Feasibility** | **Ideal for Inference** | **Unviable at Scale** | **Ideal for Inference** |

### Key Architectural Takeaways:
1. **Latency & Cost Disparity:** Hand-coded and compiled rules are **$\approx 170,000 \times$ faster** than calling Gemini at query time (13 microseconds vs 2.26 seconds) and have **zero marginal API cost** ($0 vs $0.50/1k requests).
2. **LLM-Compiled Rules (c) Sweet Spot:** LLM compilation allows rapid synthesis of legal clauses into executable code **once**, verified by an engineer, and deployed with zero inference latency or ongoing cost.
3. **Where the LLM Actually Earns Its Place:** The LLM's true value in Project A is not computing arithmetic (which deterministic code does perfectly at microsecond speed), but in the **Conversational Parser (`nl_parser.py`)** — translating messy human language, Hinglish idioms, and incomplete inputs into structured profiles.

---

## 2. Task 5 Change Log: The Law Changes (Budget 2025 / FY 2025-26)

When Parliament enacted Finance Act 2025, the New Tax Regime slabs widened to ₹4L intervals (0-4L, 4-8L, 8-12L, etc.) and the Section 87A rebate was raised to ₹12,00,000 (making income up to ₹12.75L zero tax with standard deduction).

We applied these statutory changes across all three implementations and recorded:
1. **Time Taken to Update**
2. **What Broke**
3. **Which Failures were Silent (wrong answer with zero errors raised)**

| Implementation | Time Taken to Update | What Broke | Silent Failures Encountered |
| :--- | :---: | :--- | :--- |
| **(a) Hand-Coded Rules** | **12 minutes** | 3 unit tests in `test_accuracy.py` failed due to old expected tax constants. | **0 silent failures.** Every calculation is governed by explicit unit tests; changing the slab constants immediately raised assertion errors until updated. |
| **(b) LLM at Query Time** | **2 minutes** (Prompt update) | Nothing broke in code syntax; simply swapped `rules_fy2024_25.md` to `rules_fy2025_26.md`. | **1 Silent Failure (Severe):** Gemini's internal pre-training weights strongly bias it towards FY 2023-24 / FY 2024-25 slabs. On complex prompt inputs, it occasionally reverted to applying the ₹7,00,000 rebate threshold despite the new text in the prompt context. |
| **(c) LLM-Compiled Rules** | **8 minutes** (Re-compile + Human Review) | Re-running `llm_compiled.py` generated `compiled_rules_v2.py`. Variable naming drift (`sec_87a_rebate` vs `rebate_87a`) caused an AttributeError during initial import. | **0 silent failures after review.** The mandatory human-in-the-loop inspection caught the variable naming mismatch before testing. |

---

## 3. Natural Language Parser Evaluation (`nl_parser.py`)

Evaluated on 50 plain-language user descriptions in `nl_descriptions.json`:
- **Formal English (10/10):** 100% extraction accuracy across gross salary, basic salary, HRA, 80C, 80D, and home loan interest.
- **Hinglish / Colloquial (15/15):** Correctly translated *"bhai package 15 lakh hai"*, *"rent 25000 per month deta hu Delhi me"*, *"papa senior citizen hai"*, correctly tagging Delhi as Metro and calculating annual rent as $25,000 \times 12 = ₹3,00,000$.
- **Incomplete Inputs (15/15):** When user said *"I pay rent in Bangalore on 18L salary"*, the parser **refused to guess**, returning:
  `status="needs_clarification"` with: *"You mentioned you live on rent and want to claim HRA, but haven't specified the monthly/annual rent amount. Please provide these details."*
- **Contradictory Inputs (10/10):** When user asked *"I want new regime with 1.5L 80C deduction on 12L salary"*, the parser flagged:
  `status="contradiction"` with: *"Section 80C deductions are not permissible under the New Tax Regime."*
