"""
Plain-Language Natural Language Input Parser (Project A, Task 3).

Converts free-text user descriptions (including colloquial English, Hinglish,
and incomplete inputs) into structured TaxpayerInput.

CRITICAL ARCHITECTURAL PRINCIPLE:
Must ask a clarifying question when information is ambiguous, incomplete,
or contradictory instead of guessing or hallucinating!
"""

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from project_a.common.schema import TaxpayerInput
from project_b.detectors.llm_review import (
    DEFAULT_FLASH_MODEL,
    _get_candidate_models,
    _is_model_unavailable_error,
)

load_dotenv()


@dataclass
class ParseResult:
    """Outcome of parsing a user natural language description."""
    status: str  # "complete", "needs_clarification", "contradiction"
    structured_input: TaxpayerInput | None = None
    clarifying_question: str | None = None
    contradiction_reason: str | None = None
    extracted_fields: dict[str, Any] = field(default_factory=dict)


def _make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise RuntimeError("GEMINI_API_KEY is missing.")
    return genai.Client(api_key=api_key)


def parse_natural_language_input(
    user_text: str,
    model_name: str = DEFAULT_FLASH_MODEL,
) -> ParseResult:
    """
    Parses a user input text into structured TaxpayerInput or clarifying questions.
    Handles Indian colloquial numbering (LPA, lakh, hazaar, k, cr).
    """
    if not user_text or not user_text.strip():
        return ParseResult(
            status="needs_clarification",
            clarifying_question="Please provide your annual salary to compute your income tax.",
        )

    prompt = f"""You are a strict data-extraction engine for an Indian Income Tax Calculator (FY 2024-25, AY 2025-26).
Your job is to convert the following user description into structured numeric tax inputs.

CRITICAL INSTRUCTIONS:
1. Handle Indian numbering: 'Lakh' / 'L' / 'LPA' = * 1,00,000; 'Crore' / 'Cr' = * 1,00,00,000; 'k' / 'hazaar' = * 1,000.
2. If rent is paid monthly (e.g. '25k per month', '30000 monthly'), multiply by 12 to get annual rent.
3. Metro cities for HRA in India are ONLY: Delhi, Mumbai, Kolkata, Chennai. (Bengaluru, Pune, Hyderabad, Gurgaon are Non-Metro for HRA).
4. AMBIGUITY & CLARIFICATION RULE:
   - If user claims they pay rent or want HRA, but fails to provide monthly/annual rent amount, or their basic salary is not known, you MUST set status="needs_clarification" and ask for the missing values. DO NOT GUESS.
   - If user claims insurance for family but does not specify if parents are included or if parents are senior citizens (>=60), ask for clarification.
   - If salary is completely missing or ambiguous, ask for it.
5. CONTRADICTION RULE:
   - If user asks for deductions not permissible under a regime (e.g., claiming 80C under new regime), or states basic salary > annual salary, or rent paid to self-occupied house, set status="contradiction" and explain why in "contradiction_reason".

User Text:
\"\"\"{user_text}\"\"\"

Return ONLY a JSON object matching this schema:
{{
  "status": "complete" | "needs_clarification" | "contradiction",
  "clarifying_question": "<string or null>",
  "contradiction_reason": "<string or null>",
  "fields": {{
    "annual_salary": <float or null>,
    "basic_salary": <float or null>,
    "hra_received": <float or null>,
    "rent_paid": <float or null>,
    "city_is_metro": <bool or null>,
    "section_80c": <float, default 0.0>,
    "section_80ccd1b": <float, default 0.0>,
    "section_80d_self": <float, default 0.0>,
    "section_80d_parents": <float, default 0.0>,
    "parents_are_senior_citizens": <bool, default false>,
    "home_loan_interest_self_occupied": <float, default 0.0>,
    "age": <int or null>
  }}
}}
"""

    client = _make_client()
    candidates = _get_candidate_models(model_name)
    active_model = candidates[0] if candidates else model_name

    for attempt in range(len(candidates)):
        try:
            resp = client.models.generate_content(
                model=active_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    http_options=genai_types.HttpOptions(timeout=25_000),
                ),
            )
            raw = getattr(resp, "text", "") or ""
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            status = data.get("status", "complete")
            clarifying_q = data.get("clarifying_question")
            contradiction = data.get("contradiction_reason")
            fields_data = data.get("fields", {})

            # Validate basic requirements
            if status == "complete":
                sal = fields_data.get("annual_salary")
                if sal is None or sal <= 0:
                    return ParseResult(
                        status="needs_clarification",
                        clarifying_question="Could you please specify your total annual salary or CTC?",
                    )
                tax_inp = TaxpayerInput(
                    annual_salary=float(sal),
                    basic_salary=float(fields_data["basic_salary"]) if fields_data.get("basic_salary") else None,
                    hra_received=float(fields_data["hra_received"]) if fields_data.get("hra_received") else None,
                    rent_paid=float(fields_data["rent_paid"]) if fields_data.get("rent_paid") else None,
                    city_is_metro=bool(fields_data["city_is_metro"]) if fields_data.get("city_is_metro") is not None else None,
                    section_80c=float(fields_data.get("section_80c", 0.0)),
                    section_80ccd1b=float(fields_data.get("section_80ccd1b", 0.0)),
                    section_80d_self=float(fields_data.get("section_80d_self", 0.0)),
                    section_80d_parents=float(fields_data.get("section_80d_parents", 0.0)),
                    parents_are_senior_citizens=bool(fields_data.get("parents_are_senior_citizens", False)),
                    home_loan_interest_self_occupied=float(fields_data.get("home_loan_interest_self_occupied", 0.0)),
                    age=int(fields_data["age"]) if fields_data.get("age") else None,
                )
                return ParseResult(
                    status="complete",
                    structured_input=tax_inp,
                    extracted_fields=fields_data,
                )
            elif status == "contradiction":
                return ParseResult(
                    status="contradiction",
                    contradiction_reason=contradiction or "Contradictory inputs detected.",
                    extracted_fields=fields_data,
                )
            else:
                return ParseResult(
                    status="needs_clarification",
                    clarifying_question=clarifying_q or "Please provide the missing details.",
                    extracted_fields=fields_data,
                )

        except Exception as exc:
            if _is_model_unavailable_error(exc) and attempt + 1 < len(candidates):
                active_model = candidates[attempt + 1]
                continue
            return ParseResult(
                status="needs_clarification",
                clarifying_question=f"Could not parse request ({exc}). Please state your annual salary and investments clearly.",
            )

    return ParseResult(
        status="needs_clarification",
        clarifying_question="Unable to connect to parser service. Please state your salary.",
    )
