"""
LLM Review Detectors (Project B, Task 3c).

Uses self-healing model resolution: starts with `gemini-3.8-flash` and
automatically fails over to the next candidate if `generateContent` returns a
404 / "no longer available to new users" error, caching whichever model
actually succeeds on a live call.
"""

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
import pandas as pd

load_dotenv()

FINDINGS_COLUMNS = [
    "row_id",
    "problem_type",
    "detector",
    "confidence",
    "explanation",
]

ALL_PROBLEM_TYPES = [
    "renamed_dropped_column",
    "type_change",
    "date_format_switch",
    "latlong_swap",
    "exact_duplicate",
    "near_duplicate",
    "null_spike",
    "closed_before_created",
    "zip_borough_mismatch",
    "complaint_descriptor_mismatch",
    "resolution_status_contradiction",
    "pii_in_freetext",
]

SEMANTIC_PROBLEM_TYPES = [
    "complaint_descriptor_mismatch",
    "resolution_status_contradiction",
    "pii_in_freetext",
]

STRUCTURAL_GENERIC_TYPES = [
    "renamed_dropped_column",
    "type_change",
    "date_format_switch",
    "exact_duplicate",
    "closed_before_created",
    "zip_borough_mismatch",
]

# Use gemini-3.5-flash as the default
DEFAULT_FLASH_MODEL = "gemini-3.5-flash"
_RESOLVED_MODEL_CACHE: str | None = None
_FAILED_MODELS: set[str] = set()
_CANDIDATE_MODELS_CACHE: list[str] | None = None

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

    def add(self, other: "LLMUsageStats") -> None:
        self.api_calls += other.api_calls
        self.rows_reviewed += other.rows_reviewed
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.latency_seconds += other.latency_seconds

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


def _make_client() -> genai.Client:
    """Creates a google.genai Client using the GEMINI_API_KEY env var."""
    resolved_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not resolved_key or resolved_key == "your_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is missing or not set. "
            "Set GEMINI_API_KEY in your .env file before running llm_review."
        )
    return genai.Client(api_key=resolved_key)


def _get_candidate_models(preferred: str | None = None) -> list[str]:
    """
    Builds an ordered priority list of Flash models to try, starting with
    `gemini-2.5-flash` and excluding any model in `_FAILED_MODELS`.
    """
    global _CANDIDATE_MODELS_CACHE

    priority_order = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]
    if preferred:
        clean_pref = preferred.replace("models/", "")
        if clean_pref not in priority_order:
            priority_order.insert(0, clean_pref)

    if _CANDIDATE_MODELS_CACHE is None:
        discovered = []
        try:
            client = _make_client()
            for m in client.models.list():
                name = getattr(m, "name", "") or ""
                short_name = name.replace("models/", "")
                if (
                    "flash" in short_name.lower()
                    and "tts" not in short_name.lower()
                    and "image" not in short_name.lower()
                ):
                    discovered.append(short_name)
        except Exception:
            pass

        merged = list(priority_order)
        for d in discovered:
            if d not in merged:
                merged.append(d)
        _CANDIDATE_MODELS_CACHE = merged

    return [m for m in _CANDIDATE_MODELS_CACHE if m not in _FAILED_MODELS]


def _get_gemini_model(model_name: str = DEFAULT_FLASH_MODEL, api_key: str | None = None):
    """
    Returns the resolved model name for compatibility with unit test monkeypatching.
    """
    return _resolve_model_name(model_name)


def _resolve_model_name(model_name: str = DEFAULT_FLASH_MODEL) -> str:
    """
    Returns the best available model name, checking _RESOLVED_MODEL_CACHE first,
    then picking from the candidate list.
    """
    if _RESOLVED_MODEL_CACHE and _RESOLVED_MODEL_CACHE not in _FAILED_MODELS:
        return _RESOLVED_MODEL_CACHE
    candidates = _get_candidate_models(model_name)
    return candidates[0] if candidates else model_name

def _is_model_unavailable_error(exc: Exception) -> bool:
    """
    Detects 404 / retired / unavailable / overloaded / quota exceeded errors
    that should trigger automatic failover to the next candidate model.
    """
    msg = str(exc).lower()
    triggers = (
        "404",
        "not found",
        "no longer available",
        "not available",
        "deprecated",
        " retired",
        "is not supported for generatecontent",
        "503",
        "503 unavailable",
        "high demand",
        "504",
        "deadline exceeded",
        "deadline expired",
        "429",
        "resource_exhausted",
        "quota exceeded",
    )
    return any(t in msg for t in triggers)


def _is_transient_error(exc: Exception) -> bool:
    """Detects pure JSON-parse or network blips worth a quick retry without failover."""
    msg = str(exc).lower()
    return "json" in msg or "expecting value" in msg


def _call_gemini_json(
    model, prompt: str, stats: LLMUsageStats
) -> list[dict[str, Any]]:
    """
    Executes a Gemini API call using the google.genai client with a 30-second
    timeout. `model` is accepted for API compatibility but the active model name
    is resolved from _RESOLVED_MODEL_CACHE / candidate list each call.
    Self-heals by failing over to the next candidate on 404 / unavailable errors.
    """
    global _RESOLVED_MODEL_CACHE

    # Resolve starting model name from cache or candidates
    if _RESOLVED_MODEL_CACHE and _RESOLVED_MODEL_CACHE not in _FAILED_MODELS:
        current_name = _RESOLVED_MODEL_CACHE
    else:
        candidates = _get_candidate_models()
        current_name = candidates[0] if candidates else DEFAULT_FLASH_MODEL

    client = _make_client()
    max_attempts = max(1, len(_get_candidate_models()))

    for _ in range(max_attempts):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=current_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    http_options=genai_types.HttpOptions(timeout=30_000),  # 30s in ms
                ),
            )
            stats.latency_seconds += time.perf_counter() - t0
            stats.api_calls += 1

            if _RESOLVED_MODEL_CACHE != current_name:
                _RESOLVED_MODEL_CACHE = current_name
                print(f"[llm_review] Confirmed working Gemini model: {_RESOLVED_MODEL_CACHE}")

            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta and getattr(usage_meta, "prompt_token_count", 0) > 0:
                stats.input_tokens += int(usage_meta.prompt_token_count)
                stats.output_tokens += int(getattr(usage_meta, "candidates_token_count", 0))
            else:
                stats.input_tokens += max(1, len(prompt) // 4)
                stats.output_tokens += max(1, len(getattr(response, "text", "")) // 4)

            raw_text = getattr(response, "text", "") or ""
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and isinstance(parsed.get("findings"), list):
                return parsed["findings"]
            if isinstance(parsed, list):
                return parsed
            return []

        except Exception as exc:
            stats.latency_seconds += time.perf_counter() - t0
            stats.api_calls += 1
            if _is_model_unavailable_error(exc):
                _FAILED_MODELS.add(current_name)
                if _RESOLVED_MODEL_CACHE == current_name:
                    _RESOLVED_MODEL_CACHE = None
                remaining = _get_candidate_models()
                if not remaining:
                    print(f"[llm_review] Error: all candidate Flash models failed. Last error: {exc}")
                    break
                next_name = remaining[0]
                print(
                    f"[llm_review] Model '{current_name}' unavailable ({exc}); "
                    f"failing over to '{next_name}'..."
                )
                current_name = next_name
                continue

            if _is_transient_error(exc):
                # Quick JSON/parse glitch — just log and break (data was received)
                print(f"[llm_review] Warning: Gemini call failed or returned invalid JSON: {exc}")
                break

            print(f"[llm_review] Warning: Gemini call failed or returned invalid JSON: {exc}")
            break

    return []


# ---------------------------------------------------------------------------
# 10, 11, 12. Semantic Review
# ---------------------------------------------------------------------------
def review_semantic_batch_llm(
    df: pd.DataFrame,
    problem_types: list[str] | None = None,
    batch_size: int = 25,
    model_name: str = DEFAULT_FLASH_MODEL,
    sleep_between_calls: float = 0.4,
) -> tuple[pd.DataFrame, LLMUsageStats]:
    stats = LLMUsageStats(rows_reviewed=len(df))
    if len(df) == 0:
        return _empty_findings(), stats

    allowed = [p for p in (problem_types or SEMANTIC_PROBLEM_TYPES) if p in SEMANTIC_PROBLEM_TYPES]
    if not allowed:
        return _empty_findings(), stats

    model = _resolve_model_name(model_name)
    index_lookup = {str(idx): idx for idx in df.index}
    indices = list(df.index)
    records = []
    allowed_str = ", ".join(f'"{p}"' for p in allowed)

    for start in range(0, len(indices), batch_size):
        chunk_idx = indices[start : start + batch_size]
        payload = [
            {
                "row_id": idx if isinstance(idx, (int, str)) else int(idx),
                "complaint_type": str(df.at[idx, "complaint_type"]) if "complaint_type" in df.columns else "",
                "descriptor": str(df.at[idx, "descriptor"]) if "descriptor" in df.columns else "",
                "status": str(df.at[idx, "status"]) if "status" in df.columns else "",
                "resolution_description": (
                    str(df.at[idx, "resolution_description"])
                    if "resolution_description" in df.columns
                    else ""
                ),
            }
            for idx in chunk_idx
        ]

        prompt = f"""You are a data-quality auditor for NYC 311 service requests.
Review the JSON records below and flag ONLY these problem types: [{allowed_str}].
- "complaint_descriptor_mismatch": `descriptor` is semantically incompatible with `complaint_type`.
- "resolution_status_contradiction": `resolution_description` contradicts `status` (e.g. status="Open" but resolution says case is closed, or status="Closed" but resolution says investigation has not started).
- "pii_in_freetext": `descriptor` or `resolution_description` contains private personal data (phone numbers, emails, or person names like "Maria Gonzalez", "Arthur Pendelton", "Dr. Priya Nair").

Return ONLY JSON: {{"findings": [{{"row_id": <id>, "problem_type": <type>, "confidence": <0.5-1.0>, "explanation": "<reason>"}}]}}

Records:
{json.dumps(payload, ensure_ascii=False)}"""

        for item in _call_gemini_json(model, prompt, stats):
            rid_str = str(item.get("row_id", ""))
            ptype = str(item.get("problem_type", ""))
            if rid_str in index_lookup and ptype in allowed:
                records.append({
                    "row_id": index_lookup[rid_str],
                    "problem_type": ptype,
                    "detector": "llm_review",
                    "confidence": float(item.get("confidence", 0.90)),
                    "explanation": str(item.get("explanation", "Flagged by LLM semantic review.")),
                })

        if start + batch_size < len(indices) and sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return (pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()), stats


# ---------------------------------------------------------------------------
# 1, 2, 3, 5, 8, 9. Generic Structural LLM Review Baseline
# ---------------------------------------------------------------------------
def review_structural_generic_llm(
    df: pd.DataFrame,
    problem_types: list[str] | None = None,
    batch_size: int = 25,
    model_name: str = DEFAULT_FLASH_MODEL,
    sleep_between_calls: float = 0.4,
) -> tuple[pd.DataFrame, LLMUsageStats]:
    stats = LLMUsageStats(rows_reviewed=len(df))
    if len(df) == 0:
        return _empty_findings(), stats

    allowed = [p for p in (problem_types or STRUCTURAL_GENERIC_TYPES) if p in STRUCTURAL_GENERIC_TYPES]
    if not allowed:
        return _empty_findings(), stats

    model = _resolve_model_name(model_name)
    index_lookup = {str(idx): idx for idx in df.index}
    index_lookup["-1"] = -1
    indices = list(df.index)
    records = []
    allowed_str = ", ".join(f'"{p}"' for p in allowed)

    for start in range(0, len(indices), batch_size):
        chunk_idx = indices[start : start + batch_size]
        chunk_df = df.loc[chunk_idx].copy()
        chunk_df["row_id"] = [idx if isinstance(idx, (int, str)) else int(idx) for idx in chunk_idx]
        payload = chunk_df.where(pd.notna(chunk_df), None).to_dict(orient="records")

        prompt = f"""Look at this batch of raw NYC 311 records. Does anything here look structurally wrong?
If you spot issues, classify each using ONLY one of these labels: [{allowed_str}].
For schema/column-level issues affecting the whole batch, use "row_id": -1.
Return ONLY JSON: {{"findings": [{{"row_id": <id>, "problem_type": <label>, "confidence": <0.5-1.0>, "explanation": "<reason>"}}]}}

Records:
{json.dumps(payload, default=str, ensure_ascii=False)}"""

        for item in _call_gemini_json(model, prompt, stats):
            rid_str = str(item.get("row_id", ""))
            ptype = str(item.get("problem_type", ""))
            if rid_str in index_lookup and ptype in allowed:
                records.append({
                    "row_id": index_lookup[rid_str],
                    "problem_type": ptype,
                    "detector": "llm_review",
                    "confidence": float(item.get("confidence", 0.70)),
                    "explanation": str(item.get("explanation", "Flagged by generic structural LLM review.")),
                })

        if start + batch_size < len(indices) and sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    out_df = pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()
    if not out_df.empty:
        out_df = out_df.drop_duplicates(subset=["row_id", "problem_type"], keep="first")
    return out_df, stats


# ---------------------------------------------------------------------------
# 4. Latitude/Longitude Swap LLM Review Baseline
# ---------------------------------------------------------------------------
def review_latlong_swap_llm(
    df: pd.DataFrame,
    batch_size: int = 25,
    model_name: str = DEFAULT_FLASH_MODEL,
    sleep_between_calls: float = 0.4,
) -> tuple[pd.DataFrame, LLMUsageStats]:
    stats = LLMUsageStats(rows_reviewed=len(df))
    if "latitude" not in df.columns or "longitude" not in df.columns or len(df) == 0:
        return _empty_findings(), stats

    model = _resolve_model_name(model_name)
    index_lookup = {str(idx): idx for idx in df.index}
    indices = list(df.index)
    records = []

    for start in range(0, len(indices), batch_size):
        chunk_idx = indices[start : start + batch_size]
        payload = [
            {
                "row_id": idx if isinstance(idx, (int, str)) else int(idx),
                "latitude": df.at[idx, "latitude"],
                "longitude": df.at[idx, "longitude"],
                "borough": str(df.at[idx, "borough"]) if "borough" in df.columns else "",
            }
            for idx in chunk_idx
        ]

        prompt = f"""Check whether these (latitude, longitude, borough) tuples look consistent with New York City coordinates.
Flag any row where latitude and longitude appear swapped or out of bounds as "latlong_swap".
Return ONLY JSON: {{"findings": [{{"row_id": <id>, "problem_type": "latlong_swap", "confidence": <0.5-1.0>, "explanation": "<reason>"}}]}}

Tuples:
{json.dumps(payload, default=str)}"""

        for item in _call_gemini_json(model, prompt, stats):
            rid_str = str(item.get("row_id", ""))
            if rid_str in index_lookup:
                records.append({
                    "row_id": index_lookup[rid_str],
                    "problem_type": "latlong_swap",
                    "detector": "llm_review",
                    "confidence": float(item.get("confidence", 0.85)),
                    "explanation": str(item.get("explanation", "Coordinates flagged by LLM.")),
                })

        if start + batch_size < len(indices) and sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return (pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()), stats


# ---------------------------------------------------------------------------
# 6. Near-Duplicate Within-Batch LLM Review Baseline
# ---------------------------------------------------------------------------
def review_near_duplicates_llm(
    df: pd.DataFrame,
    batch_size: int = 25,
    model_name: str = DEFAULT_FLASH_MODEL,
    sleep_between_calls: float = 0.4,
) -> tuple[pd.DataFrame, LLMUsageStats]:
    stats = LLMUsageStats(rows_reviewed=len(df))
    if len(df) < 2:
        return _empty_findings(), stats

    model = _resolve_model_name(model_name)
    index_lookup = {str(idx): idx for idx in df.index}
    indices = list(df.index)
    records = []

    for start in range(0, len(indices), batch_size):
        chunk_idx = indices[start : start + batch_size]
        payload = [
            {
                "row_id": idx if isinstance(idx, (int, str)) else int(idx),
                "unique_key": str(df.at[idx, "unique_key"]) if "unique_key" in df.columns else "",
                "created_date": str(df.at[idx, "created_date"]) if "created_date" in df.columns else "",
                "agency": str(df.at[idx, "agency"]) if "agency" in df.columns else "",
                "complaint_type": str(df.at[idx, "complaint_type"]) if "complaint_type" in df.columns else "",
                "descriptor": str(df.at[idx, "descriptor"]) if "descriptor" in df.columns else "",
            }
            for idx in chunk_idx
        ]

        prompt = f"""Review this batch of NYC 311 requests and spot likely near-duplicate pairs (the same request submitted twice with a slightly altered timestamp or minor text edit in descriptor, not verbatim identical rows).
Flag ONLY the second/later row_id of each near-duplicate pair with problem_type="near_duplicate".
Return ONLY JSON: {{"findings": [{{"row_id": <later_row_id>, "problem_type": "near_duplicate", "confidence": <0.5-1.0>, "explanation": "<reason>"}}]}}

Records:
{json.dumps(payload, default=str)}"""

        for item in _call_gemini_json(model, prompt, stats):
            rid_str = str(item.get("row_id", ""))
            if rid_str in index_lookup:
                records.append({
                    "row_id": index_lookup[rid_str],
                    "problem_type": "near_duplicate",
                    "detector": "llm_review",
                    "confidence": float(item.get("confidence", 0.80)),
                    "explanation": str(item.get("explanation", "Near-duplicate pair flagged by LLM.")),
                })

        if start + batch_size < len(indices) and sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return (pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()), stats


# ---------------------------------------------------------------------------
# 7. Null Spike LLM-on-Aggregates Review Baseline
# ---------------------------------------------------------------------------
def review_null_spike_aggregates_llm(
    df: pd.DataFrame,
    model_name: str = DEFAULT_FLASH_MODEL,
) -> tuple[pd.DataFrame, LLMUsageStats]:
    stats = LLMUsageStats(rows_reviewed=len(df))
    if "agency" not in df.columns or len(df) == 0:
        return _empty_findings(), stats

    monitored = [c for c in ("descriptor", "incident_zip", "resolution_description") if c in df.columns]
    if not monitored:
        return _empty_findings(), stats

    row_all_null = df[monitored].isna().all(axis=1)
    summary_table = []
    for agency, grp in df.groupby("agency", dropna=False):
        summary_table.append({
            "agency": str(agency),
            "total_rows": int(len(grp)),
            "null_rows": int(row_all_null.loc[grp.index].sum()),
            "null_rate": round(float(row_all_null.loc[grp.index].mean()), 4),
        })

    model = _resolve_model_name(model_name)
    prompt = f"""Below is an aggregated summary table of NYC 311 agencies and their null rates across key fields in the current batch.
Identify which agency (if any) is experiencing an anomalous null spike compared to the other agencies.
Return ONLY JSON: {{"findings": [{{"agency": "<agency_code>", "confidence": <0.5-1.0>, "explanation": "<reason>"}}]}}

Agency Summary Table:
{json.dumps(summary_table)}"""

    items = _call_gemini_json(model, prompt, stats)
    records = []
    for item in items:
        flagged_agency = str(item.get("agency", ""))
        conf = float(item.get("confidence", 0.85))
        expl = str(item.get("explanation", f"Agency '{flagged_agency}' flagged by LLM on aggregates."))
        mask = (df["agency"].astype(str) == flagged_agency) & row_all_null
        for idx in df.index[mask]:
            records.append({
                "row_id": idx,
                "problem_type": "null_spike",
                "detector": "llm_review",
                "confidence": conf,
                "explanation": expl,
            })

    return (pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()), stats


# ---------------------------------------------------------------------------
# Unified Dispatcher covering all 12 problem types for measure.py
# ---------------------------------------------------------------------------
def review_batch_with_llm(
    df: pd.DataFrame,
    problem_types: list[str] | None = None,
    batch_size: int = 25,
    max_rows: int | None = None,
    model_name: str = DEFAULT_FLASH_MODEL,
    sleep_between_calls: float = 0.4,
) -> tuple[pd.DataFrame, LLMUsageStats]:
    total_stats = LLMUsageStats()
    if len(df) == 0:
        return _empty_findings(), total_stats

    work_df = df.iloc[:max_rows] if max_rows is not None else df
    requested = problem_types or SEMANTIC_PROBLEM_TYPES
    all_findings = []

    sem_types = [p for p in requested if p in SEMANTIC_PROBLEM_TYPES]
    if sem_types:
        f_df, st = review_semantic_batch_llm(
            work_df, sem_types, batch_size=batch_size, model_name=model_name, sleep_between_calls=sleep_between_calls
        )
        total_stats.add(st)
        if not f_df.empty:
            all_findings.append(f_df)

    struct_types = [p for p in requested if p in STRUCTURAL_GENERIC_TYPES]
    if struct_types:
        f_df, st = review_structural_generic_llm(
            work_df, struct_types, batch_size=batch_size, model_name=model_name, sleep_between_calls=sleep_between_calls
        )
        total_stats.add(st)
        if not f_df.empty:
            all_findings.append(f_df)

    if "latlong_swap" in requested:
        f_df, st = review_latlong_swap_llm(
            work_df, batch_size=batch_size, model_name=model_name, sleep_between_calls=sleep_between_calls
        )
        total_stats.add(st)
        if not f_df.empty:
            all_findings.append(f_df)

    if "near_duplicate" in requested:
        f_df, st = review_near_duplicates_llm(
            work_df, batch_size=batch_size, model_name=model_name, sleep_between_calls=sleep_between_calls
        )
        total_stats.add(st)
        if not f_df.empty:
            all_findings.append(f_df)

    if "null_spike" in requested:
        f_df, st = review_null_spike_aggregates_llm(work_df, model_name=model_name)
        total_stats.add(st)
        if not f_df.empty:
            all_findings.append(f_df)

    total_stats.rows_reviewed = len(work_df)
    combined = pd.concat(all_findings, ignore_index=True) if all_findings else _empty_findings()
    return combined, total_stats


LLM_DETECTORS = {
    ptype: (lambda df, p=ptype, **kw: review_batch_with_llm(df, problem_types=[p], **kw))
    for ptype in ALL_PROBLEM_TYPES
}


def review_batch_sample(
    df: pd.DataFrame,
    sample_size: int = 50,
    seed: int = 42,
    batch_size: int = 25,
    problem_types: list[str] | None = None,
) -> pd.DataFrame:
    if len(df) == 0:
        return _empty_findings()
    sampled_df = df.sample(n=sample_size, random_state=seed) if len(df) > sample_size else df
    findings_df, _ = review_batch_with_llm(
        sampled_df, problem_types=problem_types, batch_size=batch_size
    )
    return findings_df


if __name__ == "__main__":
    if os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_API_KEY") != "your_key_here":
        model = _resolve_model_name()
        stats = LLMUsageStats()
        res = _call_gemini_json(model, 'Return ONLY JSON: {"findings": []}', stats)
        print(f"Self-healing test succeeded! Cached model: {_RESOLVED_MODEL_CACHE}, calls: {stats.api_calls}")
    else:
        print("Set GEMINI_API_KEY in .env to test live self-healing model resolution.")