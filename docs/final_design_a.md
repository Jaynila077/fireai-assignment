# Final Scale & Design — Project A (100,000 Salaried Users)

**Author:** Income Tax Engine Platform Architecture  
**Target Scale:** 100,000 active users during peak tax-filing season (June–July)  
**SLA:** Sub-second response times, 100% rupee-exact accuracy, audit-grade statutory citations  

---

## 1. Executive Summary & Core Architectural Choice

Income tax calculation for salaried individuals in India is governed by unambiguous statutory arithmetic, whereas human tax questions arrive in messy, unstructured, and colloquial language (English, Hindi, Hinglish). 

Applying an LLM to compute arithmetic live is an anti-pattern: it incurs **2.2+ seconds of latency**, costs **~$500 per 1 million queries**, and is vulnerable to catastrophic **silent hallucination** of tax slabs. Conversely, a pure hand-coded rules engine cannot interpret user text like *"mera 18 LPA package hai aur 30k rent deta hu Bengaluru me"*.

Therefore, the production architecture for 100,000 users is a **Strict 2-Stage Hybrid Architecture**:
1. **Conversational LLM Parser (`nl_parser.py`):** Translates free-text into structured, validated JSON dataclasses (`TaxpayerInput`) or generates clarifying questions if critical info is missing.
2. **Deterministic Rules Engine (`hand_coded.py` / `compiled_rules.py`):** Executes all arithmetic, deductions, slab calculations, 87A rebate, surcharge marginal relief, and cess in **13 microseconds** with **zero floating-point drift** and **100% statutory auditability**.

```
[User Chat / Mobile App]
          │
          ▼ (Plain English / Hinglish text)
┌────────────────────────────────────────────────────────┐
│ Stage 1: Conversational Parsing Gateway (Gemini Flash) │
│ • Entity extraction: Salary, Rent, HRA, 80C, 80D, NPS  │
│ • Unit normalization: LPA, Lakhs, k, monthly * 12      │
│ • Guardrail: Refuses to guess ambiguous/missing info    │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼ (Missing info?)           ▼ (Complete & Valid)
    [Ask Clarifying Question]    [Strict TaxpayerInput JSON]
                                         │
                                         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Deterministic Rules Engine (In-Memory Python) │
│ • Computes Old Regime (§ 16, 10(13A), 80C, 80D, 24b)   │
│ • Computes New Regime (§ 115BAC Slabs + 87A Marginal)  │
│ • Rounding: Section 288A (Income) & 288B (Tax)         │
│ • Statutory Clause Lineage Trace appended to result    │
│ • Latency: 13 microseconds | Cost: $0.00 compute-only  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
          [Rupee-Exact Recommendation & Explanation]
```

---

## 2. Component Labeling & Justification

| Component | Architecture Label | Latency | Cost / 1k Reqs | Justification Based on Empirical Measurements |
| :--- | :---: | :---: | :---: | :--- |
| **Input Parsing & Intent Recognition** | **LLM** | 600–900 ms | $0.12 | Handles Hinglish idioms, unstructured numbers ("18L", "25k rent"), and politely flags missing deductions or contradictory requests. Rules cannot anticipate infinite natural phrasing. |
| **Missing Information Detection** | **Rule-Guarded LLM** | < 1 ms | $0.00 | If parsed `annual_salary` is null, or `rent_paid` is claimed without amount, deterministic schema validators reject the payload immediately and prompt for clarification before running tax math. |
| **Old Regime Calculator** | **Rule** | 13 µs | $0.00 | Complex interrelated deductions (HRA Rule 2A, senior-citizen parent health insurance, ₹2L home loan limit) require rupee-exact precision. LLMs suffer roundoff and slab hallucinations. |
| **New Regime Calculator** | **Rule** | 13 µs | $0.00 | Slabs, ₹75,000 standard deduction, and Section 87A marginal relief cliff between ₹7,00,000 and ₹7,27,777 require exact threshold arithmetic. |
| **Surcharge & Marginal Relief** | **Rule** | 5 µs | $0.00 | Exact statutory threshold math at ₹50L, ₹1Cr, ₹2Cr, and ₹5Cr. LLM query-time engine occasionally miscalculates marginal surcharge caps. |
| **Regime Recommendation & Audit Lineage** | **Rule** | 2 µs | $0.00 | Comparison $T_{\text{old}} \le T_{\text{new}}$ is a single comparison. Audit trace populates citations (`Sec 115BAC`, `Sec 87A`, `Sec 288B`) with 100% legal veracity. |

---

## 3. Maintaining Correctness Through the Next Annual Budget (Finance Act)

Every February, the Ministry of Finance presents the Union Budget, altering slabs, standard deductions, or rebate limits. Maintaining correctness without disruption is achieved through **LLM Rule Compilation with Human-in-the-Loop Review**:

1. **Rule Text Extraction (`data/rules_fyXXXX_YY.md`):** Statutory amendments from the Finance Bill / Finance Act are saved as verified markdown extracts citing official Gazette clauses.
2. **Compilation Pipeline (`llm_compiled.py`):** Gemini compiles the new text into candidate code (`compiled_rules_vX.py`).
3. **Mandatory Human-in-the-Loop Review:** An engineer inspects the generated diff against `hand_coded.py`. As demonstrated in Task 5, relying solely on LLMs creates silent failures where pre-training biases override new statutory text.
4. **Automated Regression Suite (`test_accuracy.py`):** The updated engine must pass all 210 test cases plus at least 30 new edge-case tests matching the official IT Department calculator before traffic is routed.
5. **Zero-Downtime Hot Swapping:** At the start of the new Assessment Year, the system updates a single config pointer from `FY_2024_25` to `FY_2025_26`.
