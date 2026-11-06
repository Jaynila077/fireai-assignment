"""
Declarative / Schema-Check Detectors (Project B, Task 3a).

This is Tier 1 (the cheapest tier): deterministic rules that run on every row
with zero ML/LLM cost. Covers all 12 problem types so measure.py can evaluate
every (problem_type x detector_family) cell. Each function returns:
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
NYC_LAT_MIN, NYC_LAT_MAX = 40.47, 40.95
NYC_LON_MIN, NYC_LON_MAX = -74.27, -73.68

ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
ZIP_REGEX = re.compile(r"^\d{5}(-\d{4})?$")

PHONE_REGEX = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Static, incomplete allow-list of (complaint_type -> valid descriptors) for Problem 10.
STATIC_COMPLAINT_DESCRIPTOR_ALLOWLIST = {
    "Noise - Residential": {
        "Loud Music/Party",
        "Banging/Pounding",
        "Loud Talking",
        "Loud Television",
    },
    "Illegal Parking": {
        "Blocked Hydrant",
        "Posted Parking Sign Violation",
        "Blocked Sidewalk",
        "Commercial Overnight Parking",
    },
    "HEAT/HOT WATER": {
        "ENTIRE BUILDING",
        "APARTMENT ONLY",
    },
    "Street Condition": {
        "Pothole",
        "Cave-in",
        "Rough, Pitted or Cracked Roads",
    },
    "Rodent": {
        "Rat Sighting",
        "Mouse Sighting",
        "Condition Attracting Rodents",
    },
}


def _empty_findings() -> pd.DataFrame:
    """Returns an empty findings DataFrame with the standard schema."""
    return pd.DataFrame(columns=FINDINGS_COLUMNS)


def _zip_to_expected_borough(zip_str: str) -> str | None:
    """Maps a 5-digit NYC ZIP code to its expected uppercase borough name."""
    if not zip_str or len(zip_str) < 5 or not zip_str[:5].isdigit():
        return None
    prefix3 = zip_str[:3]
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
    """Checks whether any required column is missing from the DataFrame schema."""
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
    """Flags rows where incident_zip or coordinates fail strict type/format checks."""
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
    """Flags rows where created_date deviates from ISO-8601 format."""
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
    """Detects swapped or out-of-bounds coordinates using a deterministic NYC bounding box."""
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
    """Flags verbatim duplicate rows or duplicate unique_key values."""
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
# 6. Near-duplicates (Rigid exact-string + 60s rule baseline)
# ---------------------------------------------------------------------------
# Expected: very low recall on reworded near-duplicates — this is the point, included for
# completeness per Task 3 to show why fuzzy matching in statistical.py is required.
def check_near_duplicate_rigid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive rule baseline for near-duplicates: requires EXACT equality on
    (agency, complaint_type, descriptor, incident_zip) and created_date within
    60 seconds, with a different unique_key. Fails whenever descriptor has even
    a single character edit or timestamp shifts >60s.
    """
    req = {"created_date", "complaint_type", "descriptor"}
    if not req.issubset(df.columns) or len(df) < 2:
        return _empty_findings()

    work = df.copy()
    work["_ts"] = pd.to_datetime(work["created_date"], errors="coerce", format="mixed")
    group_cols = [c for c in ("agency", "complaint_type", "descriptor", "incident_zip") if c in work.columns]

    records = []
    for _, grp in work.dropna(subset=["_ts"]).groupby(group_cols, dropna=False):
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values("_ts")
        ts_diff = grp_sorted["_ts"].diff()
        # Only flag if 0 < dt <= 60s and not already an exact unique_key duplicate
        for idx in grp_sorted.index[
            (ts_diff > pd.Timedelta(seconds=0)) & (ts_diff <= pd.Timedelta(seconds=60))
        ]:
            records.append({
                "row_id": idx,
                "problem_type": "near_duplicate",
                "detector": "rule_based",
                "confidence": 0.60,
                "explanation": "Identical descriptor and complaint_type within 60 seconds.",
            })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 7. Null spike in one agency's records (Hardcoded 50% threshold baseline)
# ---------------------------------------------------------------------------
# Expected: less sensitive/more arbitrary than the statistical Z-test — included for
# completeness per Task 3 to show why relative baseline comparison outperforms static cutoffs.
def check_null_spike_fixed_threshold(
    df: pd.DataFrame, fixed_null_cutoff: float = 0.50
) -> pd.DataFrame:
    """
    Naive rule baseline for null_spike: flags an agency only if >50% of its rows
    have nulls across all monitored columns, ignoring sample size or historical
    agency baselines.
    """
    if "agency" not in df.columns or len(df) == 0:
        return _empty_findings()

    monitored = [c for c in ("descriptor", "incident_zip", "resolution_description") if c in df.columns]
    if not monitored:
        return _empty_findings()

    row_all_null = df[monitored].isna().all(axis=1)
    records = []

    for agency, grp in df.groupby("agency", dropna=False):
        null_rate = float(row_all_null.loc[grp.index].mean())
        if null_rate > fixed_null_cutoff:
            for idx in grp.index[row_all_null.loc[grp.index]]:
                records.append({
                    "row_id": idx,
                    "problem_type": "null_spike",
                    "detector": "rule_based",
                    "confidence": 0.65,
                    "explanation": (
                        f"Agency '{agency}' null rate ({null_rate:.1%}) exceeded "
                        f"hardcoded rule cutoff ({fixed_null_cutoff:.0%})."
                    ),
                })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 8. Closed date earlier than created date
# ---------------------------------------------------------------------------
def check_closed_before_created(df: pd.DataFrame) -> pd.DataFrame:
    """Flags rows where closed_date < created_date."""
    if "created_date" not in df.columns or "closed_date" not in df.columns:
        return _empty_findings()

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
    """Verifies that valid 5-digit NYC ZIP codes match the row's stated borough."""
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
# 10. Complaint Type inconsistent with Descriptor (Static allow-list baseline)
# ---------------------------------------------------------------------------
# Expected: brittle coverage and false alarms on valid unseen descriptors — included for
# completeness per Task 3 to contrast static lookup tables against statistical/LLM checks.
def check_complaint_descriptor_allowlist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive rule baseline for complaint_descriptor_mismatch: checks against a small
    hand-written allow-list of 5 common complaint types. Misses any unlisted
    complaint_type and false-alarms on legitimate descriptors not in the static set.
    """
    if "complaint_type" not in df.columns or "descriptor" not in df.columns:
        return _empty_findings()

    records = []
    for idx in df.index:
        ctype = str(df.at[idx, "complaint_type"]).strip()
        desc = str(df.at[idx, "descriptor"]).strip()
        if ctype in STATIC_COMPLAINT_DESCRIPTOR_ALLOWLIST:
            allowed_descs = STATIC_COMPLAINT_DESCRIPTOR_ALLOWLIST[ctype]
            if desc not in allowed_descs:
                records.append({
                    "row_id": idx,
                    "problem_type": "complaint_descriptor_mismatch",
                    "detector": "rule_based",
                    "confidence": 0.65,
                    "explanation": (
                        f"Descriptor '{desc}' not in static allow-list for "
                        f"complaint_type '{ctype}'."
                    ),
                })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 11. Resolution Description contradicting Status (Simple Keyword Heuristic)
# ---------------------------------------------------------------------------
def check_resolution_status_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight keyword rule for Status vs Resolution contradiction."""
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
    """Scans free-text fields for phone numbers and email addresses via regex."""
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
                break

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# Unified Registry covering all 12 problem types
# ---------------------------------------------------------------------------
RULE_DETECTORS = {
    "renamed_dropped_column": check_renamed_dropped_column,
    "type_change": check_type_change,
    "date_format_switch": check_date_format_switch,
    "latlong_swap": check_latlong_swap,
    "exact_duplicate": check_exact_duplicate,
    "near_duplicate": check_near_duplicate_rigid,
    "null_spike": check_null_spike_fixed_threshold,
    "closed_before_created": check_closed_before_created,
    "zip_borough_mismatch": check_zip_borough_mismatch,
    "complaint_descriptor_mismatch": check_complaint_descriptor_allowlist,
    "resolution_status_contradiction": check_resolution_status_keywords,
    "pii_in_freetext": check_pii_regex,
}


def run_all_rule_checks(
    df: pd.DataFrame, problem_types: list[str] | None = None
) -> pd.DataFrame:
    """Runs all (or a selected subset of) declarative rule checks on `df`."""
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

    corrupted_df, _ = inject_batch(demo_df, PROBLEM_TYPES, rate=0.10, seed=42)
    findings_df = run_all_rule_checks(corrupted_df)
    print(f"Registered rule detectors ({len(RULE_DETECTORS)}/12):", list(RULE_DETECTORS.keys()))
    print("\nRule-based findings by problem type:")
    print(findings_df["problem_type"].value_counts())