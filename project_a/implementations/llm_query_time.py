"""
(b) LLM-at-Query-Time Calculator (Project A, Task 2b).

Reads relevant statutory clauses from data/rules_fy2024_25.md dynamically,
builds a prompt with structured TaxpayerInput, and calls Gemini Flash live
to compute tax payable.

Expect lower determinism, floating-point roundoff issues, and high latency/cost
compared to (a) -- this implementation exists to measure that exact gap.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from project_a.common.schema import TaxpayerInput, TaxResult
from project_a.common import clause_refs as cr
from project_b.detectors.llm_review import (
    DEFAULT_FLASH_MODEL,
    _get_candidate_models,
    _is_model_unavailable_error,
)

load_dotenv()

RULES_PATH = Path("project_a/data/rules_fy2024_25.md")
_RULES_TEXT_CACHE: str | None = None


def _get_rules_text() -> str:
    """Caches and returns statutory rule markdown text."""
    global _RULES_TEXT_CACHE
    if _RULES_TEXT_CACHE is None:
        if RULES_PATH.exists():
            _RULES_TEXT_CACHE = RULES_PATH.read_text(encoding="utf-8")
        else:
            _RULES_TEXT_CACHE = "Income Tax Rules FY 2024-25"
    return _RULES_TEXT_CACHE


def _make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def compute_via_llm(
    inp: TaxpayerInput,
    regime: str = "new",
    model_name: str = DEFAULT_FLASH_MODEL,
) -> tuple[TaxResult, float, int]:
    """
    Computes tax via Gemini at query time.
    Returns:
      (TaxResult, latency_seconds, token_usage_count)
    """
    regime_lower = regime.strip().lower()
    assert regime_lower in ("old", "new"), "Regime must be 'old' or 'new'"

    rules_text = _get_rules_text()
    input_dict = asdict(inp)

    prompt = f"""You are an Indian Income Tax Engine for FY 2024-25 (AY 2025-26).
Compute the exact income tax for a salaried individual under the '{regime_lower}' regime according to the statutory rules below.

Statutory Rules (Official Excerpt):
\"\"\"
{rules_text}
\"\"\"

Taxpayer Input:
{json.dumps(input_dict, indent=2)}

You must perform step-by-step math and return ONLY a valid JSON object matching this exact schema:
{{
  "regime": "{regime_lower}",
  "gross_total_income": <float>,
  "total_deductions": <float>,
  "taxable_income": <float, rounded to nearest 10 per Section 288A>,
  "tax_before_cess": <float, slab tax>,
  "surcharge": <float, with marginal relief if applicable, else 0.0>,
  "cess": <float, 4% on (tax_before_cess - rebate_87a + surcharge)>,
  "rebate_87a": <float, Section 87A rebate and marginal relief if eligible, else 0.0>,
  "final_tax_payable": <float, rounded to nearest 10 per Section 288B>,
  "clause_trace": ["Sec ...", "Sec ..."]
}}
"""

    client = _make_client()
    candidates = _get_candidate_models(model_name)
    active_model = candidates[0] if candidates else model_name

    t0 = time.perf_counter()
    tokens_used = 0

    for attempt in range(len(candidates)):
        try:
            resp = client.models.generate_content(
                model=active_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    http_options=genai_types.HttpOptions(timeout=30_000),
                ),
            )
            elapsed = time.perf_counter() - t0

            if getattr(resp, "usage_metadata", None):
                tokens_used = int(getattr(resp.usage_metadata, "total_token_count", 0))
            else:
                tokens_used = max(1, (len(prompt) + len(resp.text or "")) // 4)

            raw = getattr(resp, "text", "") or ""
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            result = TaxResult(
                regime=regime_lower,
                gross_total_income=float(data.get("gross_total_income", inp.annual_salary)),
                total_deductions=float(data.get("total_deductions", 0.0)),
                taxable_income=float(data.get("taxable_income", 0.0)),
                tax_before_cess=float(data.get("tax_before_cess", 0.0)),
                surcharge=float(data.get("surcharge", 0.0)),
                cess=float(data.get("cess", 0.0)),
                rebate_87a=float(data.get("rebate_87a", 0.0)),
                final_tax_payable=float(data.get("final_tax_payable", 0.0)),
                clause_trace=list(data.get("clause_trace", [])),
            )
            return result, elapsed, tokens_used

        except Exception as exc:
            if _is_model_unavailable_error(exc) and attempt + 1 < len(candidates):
                active_model = candidates[attempt + 1]
                continue
            # Fallback mock result on total network outage to prevent crash
            elapsed = time.perf_counter() - t0
            return (
                TaxResult(
                    regime=regime_lower,
                    gross_total_income=float(inp.annual_salary),
                    total_deductions=0.0,
                    taxable_income=float(inp.annual_salary),
                    tax_before_cess=0.0,
                    surcharge=0.0,
                    cess=0.0,
                    rebate_87a=0.0,
                    final_tax_payable=0.0,
                    clause_trace=["LLM Call Failed"],
                ),
                elapsed,
                0,
            )

    return (
        TaxResult(
            regime=regime_lower,
            gross_total_income=float(inp.annual_salary),
            total_deductions=0.0,
            taxable_income=float(inp.annual_salary),
            tax_before_cess=0.0,
            surcharge=0.0,
            cess=0.0,
            rebate_87a=0.0,
            final_tax_payable=0.0,
            clause_trace=[],
        ),
        0.0,
        0,
    )
