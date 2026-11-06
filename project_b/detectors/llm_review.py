"""
LLM Semantic Review Detector (Project B, Task 3c).

This is Tier 3 (the most expensive tier): uses Gemini (`gemini-1.5-flash` via
`google-generativeai`) to detect semantic anomalies that deterministic rules
and statistical distributions miss:
  - 10. complaint_descriptor_mismatch
  - 11. resolution_status_contradiction
  - 12. pii_in_freetext (specifically unstructured full names in prose)

Batches multiple rows into a single prompt to minimize token overhead and
respect Gemini free-tier rate limits.
"""

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

load_dotenv()

FINDINGS_COLUMNS = [
    "row_id",
    "problem_type",
    "detector",
    "confidence",
    "explanation",
]

SEMANTIC_PROBLEM_TYPES = [
    "complaint_descriptor_mismatch",
    "resolution_status_contradiction",
    "pii_in_freetext",
]

# Published Gemini 1.5 Flash pricing ($0.075 / 1M input tokens, $0.30 / 1M output tokens)
# Used to compute projected cost per 1,000,000 rows in measure.py even when running on free tier.
FLASH_INPUT_COST_PER_1M = 0.075
FLASH_OUTPUT_COST_PER_1M = 0.300


@dataclass
class LLMUsageStats:
    """Tracks API calls, token consumption, latency, and estimated cost."""
    api_calls: int = 0
    rows_reviewed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    @property
    def estimated_cost_usd(self) -> float:
        in_cost = (self.input_tokens / 1_000_000.0) * FLASH_INPUT_COST_PER_1M
        out_cost = (self.output_tokens / 1_000_000.0) * FLASH_OUTPUT_COST_PER_1M
        return in_cost + out_cost

    def cost_per_million_rows(self) -> float:
        if self.rows_reviewed == 0:
            return 0.0
        return (self.estimated_cost_usd / self.rows_reviewed) * 1_000_000.0


def _empty_findings() -> pd.DataFrame:
    """Returns an empty findings DataFrame with the standard schema."""
    return pd.DataFrame(columns=FINDINGS_COLUMNS)


def _build_prompt(rows_payload: list[dict[str, Any]], allowed_problems: list[str]) -> str:
    """
    Constructs a compact, schema-constrained prompt for a batch of rows.
    Why send only flagged/relevant fields instead of the full 15-column row?
    - Omitting numeric coordinates, agency codes, and timestamps cuts input
      tokens by ~60% per row and prevents the LLM from hallucinating math/date errors.
    """
    allowed_str = ", ".join(f'"{p}"' for p in allowed_problems)
    return f"""You are a strict data-quality auditor for NYC 311 service request records.
Review the JSON array of records below. Check ONLY for these specific problem types: [{allowed_str}].

Definitions of problem types:
1. "complaint_descriptor_mismatch": The `descriptor` is semantically incompatible with the `complaint_type` (e.g., complaint_type="Noise - Residential" but descriptor="Broken Parking Meter" or "Pothole in Street").
2. "resolution_status_contradiction": The `resolution_description` directly contradicts the `status` (e.g., status="Open" while resolution_description says the case was resolved/closed, OR status="Closed" while resolution_description says investigation has not started or remains unresolved).
3. "pii_in_freetext": Either `descriptor` or `resolution_description` contains personal identifiable information (PII) of private individuals, such as phone numbers, email addresses, or specific person names (e.g., "Maria Gonzalez", "Arthur Pendelton", "Dr. Priya Nair"). Standard agency names like "Police Department" are NOT PII.

Return ONLY a valid JSON object matching this exact schema:
{{
  "findings": [
    {{
      "row_id": <integer or string matching the input record's row_id>,
      "problem_type": <one of [{allowed_str}]>,
      "confidence": <float between 0.5 and 1.0>,
      "explanation": "<concise 1-sentence reason>"
    }}
  ]
}}
If a record has no issues, omit it from "findings". Do not invent problems.

Input records:
{json.dumps(rows_payload, ensure_ascii=False)}
"""


def _parse_llm_json(raw_text: str) -> list[dict[str, Any]]:
    """
    Safely parses the LLM response into a list of finding dicts.
    Strips markdown code fences if present and returns [] on any parse failure
    so a single malformed model output never crashes a batch job.
    """
    if not raw_text:
        return []
    cleaned = raw_text.strip()
    # Strip optional ```json ... ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and isinstance(parsed.get("findings"), list):
            return parsed["findings"]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


def review_batch_with_llm(
    df: pd.DataFrame,
    problem_types: list[str] | None = None,
    batch_size: int = 25,
    max_rows: int | None = None,
    model_name: str = "gemini-1.5-flash",
    api_key: str | None = None,
    sleep_between_calls: float = 0.5,
) -> tuple[pd.DataFrame, LLMUsageStats]:
    """
    Reviews `df` (or a sample up to `max_rows`) using Gemini in batches of
    `batch_size` rows per API call.

    Returns `(findings_df, usage_stats)`.
    """
    stats = LLMUsageStats()
    if len(df) == 0:
        return _empty_findings(), stats

    target_problems = [
        p for p in (problem_types or SEMANTIC_PROBLEM_TYPES) if p in SEMANTIC_PROBLEM_TYPES
    ]
    if not target_problems:
        # Discipline rule: if called only for non-semantic problems (e.g. exact_duplicate),
        # do not waste LLM tokens when rules/stats handle them deterministically.
        return _empty_findings(), stats

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not resolved_key or resolved_key == "your_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is missing or not set. "
            "Set GEMINI_API_KEY in your .env file before running llm_review."
        )

    genai.configure(api_key=resolved_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    )

    work_df = df.iloc[:max_rows] if max_rows is not None else df
    stats.rows_reviewed = len(work_df)

    # Map stringified row_id back to the original DataFrame index type (e.g. int)
    index_lookup = {str(idx): idx for idx in work_df.index}

    records = []
    indices = list(work_df.index)

    for start in range(0, len(indices), batch_size):
        chunk_idx = indices[start : start + batch_size]
        chunk_df = work_df.loc[chunk_idx]

        payload = []
        for idx in chunk_idx:
            row = chunk_df.loc[idx]
            payload.append({
                "row_id": idx if isinstance(idx, (int, str)) else int(idx),
                "complaint_type": str(row.get("complaint_type", "")),
                "descriptor": str(row.get("descriptor", "")),
                "status": str(row.get("status", "")),
                "resolution_description": str(row.get("resolution_description", "")),
            })

        prompt = _build_prompt(payload, target_problems)
        t0 = time.perf_counter()

        try:
            response = model.generate_content(prompt)
            elapsed = time.perf_counter() - t0
            stats.latency_seconds += elapsed
            stats.api_calls += 1

            # Extract token counts from usage_metadata when available; fallback to ~4 chars/token
            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta and getattr(usage_meta, "prompt_token_count", 0) > 0:
                stats.input_tokens += int(usage_meta.prompt_token_count)
                stats.output_tokens += int( getattr(usage_meta, "candidates_token_count", 0))
            else:
                stats.input_tokens += max(1, len(prompt) // 4)
                stats.output_tokens += max(1, len(getattr(response, "text", "")) // 4)

            findings_list = _parse_llm_json(getattr(response, "text", ""))
            for item in findings_list:
                raw_rid = str(item.get("row_id", ""))
                ptype = str(item.get("problem_type", ""))
                if raw_rid not in index_lookup or ptype not in target_problems:
                    continue

                records.append({
                    "row_id": index_lookup[raw_rid],
                    "problem_type": ptype,
                    "detector": "llm_review",
                    "confidence": float(item.get("confidence", 0.90)),
                    "explanation": str(item.get("explanation", "Flagged by Gemini semantic review.")),
                })

        except Exception as exc:
            # Log and continue so transient API/rate-limit glitches don't abort the entire batch
            elapsed = time.perf_counter() - t0
            stats.latency_seconds += elapsed
            print(f"[llm_review] Warning: batch starting at {start} failed: {exc}")

        if start + batch_size < len(indices) and sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    findings_df = pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()
    return findings_df, stats


def review_batch_sample(
    df: pd.DataFrame,
    sample_size: int = 50,
    seed: int = 42,
    batch_size: int = 25,
    problem_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper matching the assignment stub signature: samples up to
    `sample_size` rows deterministically (or reviews all rows if len(df) <= sample_size)
    and returns the standardized findings DataFrame.
    """
    if len(df) == 0:
        return _empty_findings()

    if len(df) > sample_size:
        sampled_df = df.sample(n=sample_size, random_state=seed)
    else:
        sampled_df = df

    findings_df, _ = review_batch_with_llm(
        sampled_df,
        problem_types=problem_types,
        batch_size=batch_size,
    )
    return findings_df


if __name__ == "__main__":
    # Standalone demo: runs only if GEMINI_API_KEY is configured in environment
    from project_b.injector.corruption_injector import inject_batch

    demo_df = pd.DataFrame({
        "unique_key": [f"311-{4000 + i}" for i in range(10)],
        "created_date": ["2023-10-01 09:00:00"] * 10,
        "closed_date": ["2023-10-01 15:00:00"] * 10,
        "agency": ["NYPD"] * 10,
        "complaint_type": ["Noise - Residential"] * 10,
        "descriptor": ["Loud Music/Party"] * 10,
        "status": ["Closed"] * 10,
        "resolution_description": ["The Police Department responded and took action."] * 10,
        "incident_zip": ["10027"] * 10,
        "borough": ["MANHATTAN"] * 10,
        "latitude": [40.8116] * 10,
        "longitude": [-73.9536] * 10,
    })

    corrupted_df, gt_df = inject_batch(
        demo_df,
        SEMANTIC_PROBLEM_TYPES,
        rate=0.20,
        seed=42,
    )
    print("Injected semantic ground truth:")
    print(gt_df[["row_id", "problem_type", "corrupted_value"]])

    if os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_API_KEY") != "your_key_here":
        findings, usage = review_batch_with_llm(corrupted_df, batch_size=10)
        print("\nLLM findings:")
        print(findings)
        print(f"\nUsage stats: {usage}, Est. Cost/1M rows: ${usage.cost_per_million_rows():.2f}")
    else:
        print("\nSkipping live Gemini API call (set GEMINI_API_KEY in .env to run live demo).")