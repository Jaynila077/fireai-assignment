# AI Usage & Workflow Notes

I used AI tools throughout this take-home assignment both as system components inside the pipeline (Gemini API for live queries and data review) and as coding assistants (Antigravity / Cursor-style assistant for boilerplate and test data generation). 

Here is an honest breakdown of what worked, what failed, and the time saved.

---

## 1. Tools Used

- **Google Gemini 1.5 / 2.0 Flash (`gemini-2.5-flash` / `gemini-3.5-flash-lite` via `google.genai` SDK):** 
  - Used inside the actual deliverables for Project A (query-time tax calculation and natural language parser) and Project B (Tier 3 semantic detector for subtle NYC 311 contradictions).
  - Also used once as an offline "compiler" to translate statutory tax rules from Markdown into standalone Python code (`compiled_rules_v1.py`).
- **Antigravity Coding Assistant:** 
  - Used for rapid scaffolding, generating combinatorial test cases, drafting regex patterns for NYC 311 data, and automating benchmark measurement scripts.

---

## 2. Workflows & Prompts That Actually Worked

### What Worked Really Well:
1. **Generating Combinatorial Test Cases:**  
   Writing 200+ tax test cases manually is mind-numbing. Prompting the model with the exact JSON schema and asking for specific cross-sections (e.g., "give me 30 salary profiles around the ₹7L rebate threshold, varying 80C and 80D senior citizen deductions") saved hours. I cross-checked 30 of these against the official Income Tax Department calculator to verify ground truth.
2. **Drafting Natural Language & Hinglish Descriptions:**  
   Generating 50 messy, colloquial taxpayer queries (including Hinglish phrases like *"arre bhai HRA deduct karo metro city Bangalore"* and intentionally contradictory statements) was fast and gave a realistic test set.
3. **Structured JSON Extraction with System Constraints:**  
   For the natural language parser (`nl_parser.py`), few-shot prompts with strict schema definitions worked reliably—provided I explicitly instructed the model *not* to guess missing parameters.
4. **Offline Rule Compilation (Task 2c):**  
   Prompting the LLM with the raw statutory text of the Finance Act and asking for a clean, deterministic Python function produced a solid draft in seconds. It needed human cleanup (fixing variable names and edge-case rebate rounding), but as an offline compiler, it was effective.

---

## 3. Where the AI Was Wrong & How I Caught It

Here are three specific cases where the AI generated incorrect code or logic, and how I caught and fixed it.

### Example 1: Pre-training bias silently overrode 2025 Budget tax rules (Project A, Task 5)
- **The Issue:** When testing Budget 2025 changes, I updated the prompt with the new FY 2025-26 rules (raising the Section 87A rebate ceiling to ₹12 Lakhs). But when I ran a query with a taxable income of ₹9,50,000, Gemini calculated ₹45,000 tax plus cess. It completely ignored the new ₹12L rebate clause provided directly in the prompt context.
- **Why it happened:** The model's pre-training data was saturated with thousands of articles explaining the FY 2023-24 / FY 2024-25 ₹7 Lakh rebate limit. Its internal priors overpowered the prompt text.
- **How I caught it:** My regression test suite (`test_accuracy.py`) had a hard ground-truth check expecting ₹0 tax for ₹9.5L income under FY 2025-26. The assertion blew up immediately. This made it crystal clear why financial engines shouldn't rely on raw LLMs at query time.

### Example 2: The LLM was "too helpful" and invented missing rent amounts (Project A, Task 3)
- **The Issue:** In `nl_parser.py`, when testing incomplete prompts like *"I make 18 LPA and pay rent in Bangalore"*, the model happily filled in `monthly_rent: 25000` and marked the status as `complete`.
- **Why it happened:** LLMs hate saying "I don't know." They are tuned to give complete answers and make plausible assumptions.
- **How I caught it:** I ran my 15 "incomplete description" test cases through the parser. The test asserted that incomplete inputs must return `status="needs_clarification"` with `missing_fields`. Because the model hallucinated ₹25k, the assertion failed. I fixed it by tightening the system prompt with an explicit guardrail: *"Never assume or hallucinate unstated financial figures; if rent or basic salary is missing, you MUST return needs_clarification"*.

### Example 3: Outdated SDK calls and hanging Windows timeouts (Project B, Task 3c)
- **The Issue:** The initial boilerplate drafted by the assistant used the older `google.generativeai` package and simple `time.sleep()`. On Windows, network calls would occasionally stall or drop with 504 deadline errors, and the script would hang indefinitely or misinterpret transient errors as permanent model failures.
- **Why it happened:** The assistant used outdated code snippets from older documentation that didn't handle the new Google GenAI SDK's `HttpOptions` timeouts properly.
- **How I caught it:** Pytest hung on `test_detectors.py` during live integration runs. I checked the traceback, killed the process, and migrated the code to the modern `google.genai` SDK with explicit 30-second client timeouts (`genai_types.HttpOptions(timeout=30_000)`). I also added a model failover list (`gemini-2.5-flash`, `gemini-3.5-flash-lite`, etc.) so transient rate limits or quota drops didn't crash the pipeline.

---

## 4. Rough Estimate of Time Saved

- **Test case synthesis:** ~5–6 hours saved. Generating 210 test cases, edge cases, and 50 realistic Hinglish prompts by hand would have taken almost an entire work day.
- **Detector boilerplate & regexes:** ~3–4 hours saved. Generating baseline regexes for PII, timestamp parsing, and NYC coordinate bounding boxes.
- **Architecture design docs:** ~2 hours saved on formatting and outlining `final_design_a.md` and `final_design_b.md`.
- **Time lost debugging AI mistakes:** -2 hours spent debugging LLM tax calculation errors, fixing hallucinated inputs in the parser, and diagnosing deprecated SDK timeout issues.

**Net time saved:** ~8 to 10 hours overall.
