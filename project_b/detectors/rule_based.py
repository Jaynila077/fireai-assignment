"""
Declarative / Schema-Check Detectors (Project B, Task 3a).

This is Tier 1 (the cheapest tier): deterministic rules that run on every row
with zero ML/LLM cost. Each function returns a standardized findings DataFrame:
[row_id, problem_type, detector, confidence, explanation].
"""

import re
import numpy as np
import pandas as pd

FINDINGS_COLUMNS = [
    "row_id",
    "problem_type",
    "detector",
    "confidence",
    "explanation",
]

# Expected core columns in a clean NYC 311 batch.
EXPECTED_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "complaint_type",
    "descriptor",
    "status",
    "resolution_description",
    "incident_zip",
    "borough",
    "latitude",
    "longitude",
]

# Tight bounding box around the 5 boroughs of New York City.
# Swapping lat (~40.7) and lon (~-73.9) immediately falls outside this box.
NYC_LAT_MIN, NYC_LAT_MAX = 40.47, 40.95
NYC_LON_MIN, NYC_LON_MAX = -74.27, -73.68

# Strict ISO-8601 timestamp pattern expected from Socrata (YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS).
ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")

# Strict 5-digit US ZIP code (or 5+4 format). Strings like "10027.0" fail this check.
ZIP_REGEX = re.compile(r"^\d{5}(-\d{4})?$")

# Compiled regexes for structured PII (North American phone numbers and standard emails).
PHONE_REGEX = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _empty_findings() -> pd.DataFrame:
    """Returns an empty findings DataFrame with the standard schema."""
    return pd.DataFrame(columns=FINDINGS_COLUMNS)


def _zip_to_expected_borough(zip_str: str) -> str | None:
    """
    Maps a 5-digit NYC ZIP code to its expected uppercase borough name using
    USPS 3-digit prefix allocation rules for NYC. Returns None if not a known
    NYC prefix.
    """
    if not zip_str or len(zip_str) < 5 or not zip_str[:5].isdigit():
        return None
    z5 = zip_str[:5]
    prefix3 = z5[:3]

    # USPS 3-digit prefix assignments for New York City boroughs:
    if prefix3 in ("100", "101", "102"):
        return "MANHATTAN"
    if prefix3 == "103":
        return "STATEN ISLAND"
    if prefix3 == "104":
        return "BRONX"
    if prefix3 == "112":
        return "BROOKLYN"
    if prefix3 in ("110", "111", "113", "114", "116"):
        return "QUEENS"
    return None


# ---------------------------------------------------------------------------
# 1. Renamed or dropped column
# ---------------------------------------------------------------------------
def check_renamed_dropped_column(
    df: pd.DataFrame, expected_columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Checks whether any required column is missing from the DataFrame schema.
    Logs row_id = -1 to match the batch-level ground-truth convention in the injector.
    """
    expected = expected_columns or EXPECTED_COLUMNS
    present = set(df.columns)
    missing = [col for col in expected if col not in present]
    unexpected = [col for col in df.columns if col not in expected]

    records = []
    for col in missing:
        hint = f" (found unexpected columns: {unexpected})" if unexpected else ""
        records.append({
            "row_id": -1,
            "problem_type": "renamed_dropped_column",
            "detector": "rule_based",
            "confidence": 1.0,
            "explanation": f"Required column '{col}' is missing from batch schema{hint}.",
        })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 2. Column type change
# ---------------------------------------------------------------------------
def check_type_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags rows where incident_zip fails strict 5-digit ZIP formatting (e.g.
    float-cast strings like '10027.0' or words like 'ten thousand twenty seven')
    or where latitude/longitude cannot be coerced to float.
    """
    records = []

    if "incident_zip" in df.columns:
        non_null_zip = df["incident_zip"].notna()
        zip_str = df.loc[non_null_zip, "incident_zip"].astype(str).str.strip()
        bad_zip_mask = ~zip_str.str.match(ZIP_REGEX)
        for idx, val in zip_str[bad_zip_mask].items():
            records.append({
                "row_id": idx,
                "problem_type": "type_change",
                "detector": "rule_based",
                "confidence": 1.0,
                "explanation": f"incident_zip '{val}' is not a valid 5-digit ZIP string.",
            })

    for coord_col in ("latitude", "longitude"):
        if coord_col in df.columns:
            non_null = df[coord_col].notna()
            coerced = pd.to_numeric(df.loc[non_null, coord_col], errors="coerce")
            bad_coord_mask = coerced.isna()
            for idx in coerced[bad_coord_mask].index:
                val = df.at[idx, coord_col]
                records.append({
                    "row_id": idx,
                    "problem_type": "type_change",
                    "detector": "rule_based",
                    "confidence": 1.0,
                    "explanation": f"{coord_col} value '{val}' cannot be parsed as float.",
                })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 3. Date format switch mid-batch
# ---------------------------------------------------------------------------
def check_date_format_switch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags rows where created_date deviates from the expected ISO-8601 format
    (YYYY-MM-DD HH:MM:SS), catching mid-batch flips to DD/MM/YYYY or text dates.
    """
    if "created_date" not in df.columns:
        return _empty_findings()

    non_null = df["created_date"].notna()
    date_str = df.loc[non_null, "created_date"].astype(str).str.strip()
    bad_fmt_mask = ~date_str.str.match(ISO_DATE_REGEX)

    records = [
        {
            "row_id": idx,
            "problem_type": "date_format_switch",
            "detector": "rule_based",
            "confidence": 1.0,
            "explanation": f"created_date '{val}' does not match ISO format YYYY-MM-DD HH:MM:SS.",
        }
        for idx, val in date_str[bad_fmt_mask].items()
    ]
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 4. Latitude/longitude swapped (via NYC bounding-box rule)
# ---------------------------------------------------------------------------
def check_latlong_swap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects swapped or out-of-bounds coordinates using a deterministic bounding
    box around NYC. Specifically notes when latitude falls in NYC's longitude
    range and vice versa.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return _empty_findings()

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid_num = lat.notna() & lon.notna()

    out_of_box = valid_num & (
        (lat < NYC_LAT_MIN)
        | (lat > NYC_LAT_MAX)
        | (lon < NYC_LON_MIN)
        | (lon > NYC_LON_MAX)
    )

    records = []
    for idx in df.index[out_of_box]:
        la, lo = lat.at[idx], lon.at[idx]
        # Check if swapping them places the point right back inside NYC
        is_exact_swap = (NYC_LAT_MIN <= lo <= NYC_LAT_MAX) and (NYC_LON_MIN <= la <= NYC_LON_MAX)
        detail = "swapped lat/lon values" if is_exact_swap else "outside NYC bounding box"
        records.append({
            "row_id": idx,
            "problem_type": "latlong_swap",
            "detector": "rule_based",
            "confidence": 1.0 if is_exact_swap else 0.9,
            "explanation": f"Coordinates ({la}, {lo}) are {detail}.",
        })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 5. Exact duplicates
# ---------------------------------------------------------------------------
def check_exact_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags duplicate rows using `keep='first'` so only the second and subsequent
    cloned occurrences are flagged (matching the injector's ground-truth indices).
    Checks both full-row equality and duplicate unique_key values.
    """
    if len(df) == 0:
        return _empty_findings()

    dup_mask = df.duplicated(keep="first")
    if "unique_key" in df.columns:
        dup_mask = dup_mask | df.duplicated(subset=["unique_key"], keep="first")

    records = [
        {
            "row_id": idx,
            "problem_type": "exact_duplicate",
            "detector": "rule_based",
            "confidence": 1.0,
            "explanation": (
                f"Exact duplicate of an earlier row in the batch "
                f"(unique_key={df.at[idx, 'unique_key'] if 'unique_key' in df.columns else 'N/A'})."
            ),
        }
        for idx in df.index[dup_mask]
    ]
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 8. Closed date earlier than created date
# ---------------------------------------------------------------------------
def check_closed_before_created(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags rows where closed_date < created_date when both timestamps are present
    and parseable.
    """
    if "created_date" not in df.columns or "closed_date" not in df.columns:
        return _empty_findings()

    # Use format='ISO8601' or fallback coercion so valid timestamps compare cleanly
    created = pd.to_datetime(df["created_date"], errors="coerce", format="mixed")
    closed = pd.to_datetime(df["closed_date"], errors="coerce", format="mixed")

    inverted_mask = created.notna() & closed.notna() & (closed < created)

    records = [
        {
            "row_id": idx,
            "problem_type": "closed_before_created",
            "detector": "rule_based",
            "confidence": 1.0,
            "explanation": (
                f"closed_date ({df.at[idx, 'closed_date']}) precedes "
                f"created_date ({df.at[idx, 'created_date']})."
            ),
        }
        for idx in df.index[inverted_mask]
    ]
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 9. ZIP code inconsistent with borough
# ---------------------------------------------------------------------------
def check_zip_borough_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifies that valid 5-digit NYC ZIP codes match the row's stated borough
    using the static USPS NYC prefix lookup table.
    """
    if "incident_zip" not in df.columns or "borough" not in df.columns:
        return _empty_findings()

    records = []
    for idx in df.index:
        raw_zip = df.at[idx, "incident_zip"]
        raw_borough = df.at[idx, "borough"]
        if pd.isna(raw_zip) or pd.isna(raw_borough):
            continue

        zip_str = str(raw_zip).strip()
        borough_str = str(raw_borough).strip().upper()
        if borough_str in ("UNSPECIFIED", ""):
            continue

        expected_borough = _zip_to_expected_borough(zip_str)
        # Only flag if the ZIP is a valid 5-digit NYC ZIP mapped to a DIFFERENT borough
        # (malformed ZIP strings like '10027.0' are handled by check_type_change).
        if expected_borough is not None and expected_borough != borough_str:
            records.append({
                "row_id": idx,
                "problem_type": "zip_borough_mismatch",
                "detector": "rule_based",
                "confidence": 1.0,
                "explanation": (
                    f"incident_zip '{zip_str}' belongs to {expected_borough}, "
                    f"but borough is '{borough_str}'."
                ),
            })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 11. Resolution Description contradicting Status (Simple Keyword Heuristic)
# ---------------------------------------------------------------------------
def check_resolution_status_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    A lightweight keyword rule for Status vs Resolution contradiction. Included
    so we can measure how far cheap keyword rules get vs. LLM semantic review.
    """
    if "status" not in df.columns or "resolution_description" not in df.columns:
        return _empty_findings()

    records = []
    for idx in df.index:
        status = str(df.at[idx, "status"]).strip().upper()
        res = str(df.at[idx, "resolution_description"]).strip()
        res_lower = res.lower()
        if not res or res == "nan":
            continue

        if status == "OPEN" and ("case closed" in res_lower or "closed the ticket" in res_lower):
            records.append({
                "row_id": idx,
                "problem_type": "resolution_status_contradiction",
                "detector": "rule_based",
                "confidence": 0.75,
                "explanation": "Status is 'Open' but resolution_description contains explicit closure phrase.",
            })
        elif status == "CLOSED" and (
            "remains active and unresolved" in res_lower
            or "investigation has not started" in res_lower
        ):
            records.append({
                "row_id": idx,
                "problem_type": "resolution_status_contradiction",
                "detector": "rule_based",
                "confidence": 0.75,
                "explanation": "Status is 'Closed' but resolution_description states issue is unresolved.",
            })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 12. Personal data (PII) in free text (Regex check for phones & emails)
# ---------------------------------------------------------------------------
def check_pii_regex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans free-text fields (resolution_description and descriptor) for phone
    numbers and email addresses using regular expressions. Fast and free, but
    cannot catch unstructured person names without titles/patterns.
    """
    text_cols = [c for c in ("resolution_description", "descriptor") if c in df.columns]
    if not text_cols:
        return _empty_findings()

    records = []
    for idx in df.index:
        for col in text_cols:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            text = str(val)
            has_phone = bool(PHONE_REGEX.search(text))
            has_email = bool(EMAIL_REGEX.search(text))
            if has_phone or has_email:
                matched_types = []
                if has_phone:
                    matched_types.append("phone number")
                if has_email:
                    matched_types.append("email address")
                records.append({
                    "row_id": idx,
                    "problem_type": "pii_in_freetext",
                    "detector": "rule_based",
                    "confidence": 0.95,
                    "explanation": f"Detected {', '.join(matched_types)} in '{col}' via regex.",
                })
                break  # Avoid double-counting the same row if both columns have PII

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# Unified Runner for All Rule-Based Checks
# ---------------------------------------------------------------------------
RULE_DETECTORS = {
    "renamed_dropped_column": check_renamed_dropped_column,
    "type_change": check_type_change,
    "date_format_switch": check_date_format_switch,
    "latlong_swap": check_latlong_swap,
    "exact_duplicate": check_exact_duplicate,
    "closed_before_created": check_closed_before_created,
    "zip_borough_mismatch": check_zip_borough_mismatch,
    "resolution_status_contradiction": check_resolution_status_keywords,
    "pii_in_freetext": check_pii_regex,
}


def run_all_rule_checks(
    df: pd.DataFrame, problem_types: list[str] | None = None
) -> pd.DataFrame:
    """
    Runs all (or a selected subset of) declarative rule checks on `df` and
    returns a single concatenated findings DataFrame.
    """
    target_types = problem_types or list(RULE_DETECTORS.keys())
    results = []

    for ptype in target_types:
        detector_fn = RULE_DETECTORS.get(ptype)
        if detector_fn is not None:
            res = detector_fn(df)
            if not res.empty:
                results.append(res)

    return pd.concat(results, ignore_index=True) if results else _empty_findings()


if __name__ == "__main__":
    # Quick standalone verification against corruption_injector
    from project_b.injector.corruption_injector import PROBLEM_TYPES, inject_batch

    demo_df = pd.DataFrame({
        "unique_key": [f"311-{2000 + i}" for i in range(20)],
        "created_date": ["2023-08-15 10:00:00"] * 20,
        "closed_date": ["2023-08-15 16:00:00"] * 20,
        "agency": ["NYPD"] * 12 + ["DOT"] * 8,
        "agency_name": ["New York City Police Department"] * 12 + ["Department of Transportation"] * 8,
        "complaint_type": ["Noise - Residential"] * 12 + ["Street Condition"] * 8,
        "descriptor": ["Loud Music/Party"] * 12 + ["Pothole"] * 8,
        "status": ["Closed"] * 20,
        "resolution_description": ["The Police Department responded and took action."] * 20,
        "incident_zip": ["10027"] * 12 + ["11201"] * 8,
        "borough": ["MANHATTAN"] * 12 + ["BROOKLYN"] * 8,
        "latitude": [40.8116] * 12 + [40.6943] * 8,
        "longitude": [-73.9536] * 12 + [-73.9903] * 8,
    })

    corrupted_df, gt_df = inject_batch(demo_df, PROBLEM_TYPES, rate=0.10, seed=42)
    findings_df = run_all_rule_checks(corrupted_df)
    print("Rule-based detector findings by problem type:")
    print(findings_df["problem_type"].value_counts())