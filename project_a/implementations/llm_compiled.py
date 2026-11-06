"""
(c) LLM-Compiled Rules (Project A, Task 2c).

Run ONCE: Prompts Gemini to convert statutory rules markdown (data/rules_fy2024_25.md)
into an executable Python module saved to compiled_rules/compiled_rules_v1.py.

Human-in-the-loop review: Every line of the generated compiled rules is verified
before being committed and imported for inference. At query time, compiled_rules_v1.py
runs as pure deterministic Python with ZERO LLM API calls.
"""

from pathlib import Path
import re
from project_b.detectors.llm_review import (
    DEFAULT_FLASH_MODEL,
    _get_candidate_models,
    _make_client,
)
from google.genai import types as genai_types

RULES_PATH = Path("project_a/data/rules_fy2024_25.md")
DEFAULT_OUTPUT_PATH = Path("project_a/compiled_rules/compiled_rules_v1.py")


def compile_rules_from_text(
    rules_md_path: Path = RULES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_FLASH_MODEL,
) -> str:
    """
    Reads rule markdown text, prompts Gemini to compile executable Python functions:
      - compute_old_regime(inp: TaxpayerInput) -> TaxResult
      - compute_new_regime(inp: TaxpayerInput) -> TaxResult
      - recommend_regime(inp: TaxpayerInput) -> TaxResult
    Writes code to output_path and returns the source code string.
    """
    if not rules_md_path.exists():
        raise FileNotFoundError(f"Rules text not found: {rules_md_path}")

    rules_text = rules_md_path.read_text(encoding="utf-8")

    prompt = f"""You are a compiler that translates legal statutory tax text into clean, executable Python code.
Translate the following Indian Income Tax Rules for FY 2024-25 (AY 2025-26) into a standalone Python module.

Statutory Rules Text:
\"\"\"
{rules_text}
\"\"\"

Requirements:
1. Must import:
   from decimal import Decimal, ROUND_HALF_UP
   from project_a.common.schema import TaxpayerInput, TaxResult
   from project_a.common import clause_refs as cr
2. Must implement:
   - compute_old_regime(inp: TaxpayerInput) -> TaxResult
   - compute_new_regime(inp: TaxpayerInput) -> TaxResult
   - recommend_regime(inp: TaxpayerInput) -> TaxResult
3. Must adhere to:
   - Rounding off under Section 288A (taxable income to nearest 10) and Section 288B (tax payable to nearest 10).
   - Old regime standard deduction ₹50,000, New regime standard deduction ₹75,000.
   - New regime rebate up to ₹7,00,000 plus marginal relief up to ₹7,27,777.
   - Surcharge with marginal relief and 4% Health & Education Cess.
   - Population of clause_trace on every TaxResult.
4. Output ONLY valid, clean Python code without markdown ticks or explanation.
"""

    client = _make_client()
    candidates = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"]

    for active_model in candidates:
        try:
            print(f"[llm_compiled] Attempting compilation via model: {active_model}...")
            resp = client.models.generate_content(
                model=active_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    http_options=genai_types.HttpOptions(timeout=60_000),
                ),
            )
            raw_code = getattr(resp, "text", "") or ""
            if raw_code.strip():
                cleaned_code = re.sub(r"^```(?:python)?\s*", "", raw_code.strip())
                cleaned_code = re.sub(r"\s*```$", "", cleaned_code)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(cleaned_code, encoding="utf-8")
                print(f"[llm_compiled] Compiled rules written to {output_path} ({len(cleaned_code)} chars)")
                return cleaned_code
        except Exception as exc:
            print(f"[llm_compiled] Model {active_model} failed: {exc}")
            continue

    raise RuntimeError("All candidate models failed compilation.")


if __name__ == "__main__":
    compile_rules_from_text()
