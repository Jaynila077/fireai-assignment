"""
Corruption Injector for NYC 311 Service Requests (Project B, Task 2).

Takes a clean DataFrame and injects 12 realistic data-quality problems at a
controlled rate. Always works on a copy of the input DataFrame and uses a
local numpy Generator so runs with the same seed are 100% deterministic.
"""

from typing import Callable
import numpy as np
import pandas as pd

PROBLEM_TYPES = [
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

GROUND_TRUTH_COLUMNS = [
    "row_id",
    "unique_key",
    "problem_type",
    "column_touched",
    "original_value",
    "corrupted_value",
]

# Real NYC ZIP codes mapped by borough so mismatch injection uses plausible NYC ZIPs
# rather than nonsense strings that any regex would trivially catch.
NYC_BOROUGH_ZIPS = {
    "MANHATTAN": ["10001", "10002", "10011", "10025", "10027", "10036"],
    "BROOKLYN": ["11201", "11206", "11211", "11215", "11226", "11238"],
    "QUEENS": ["11101", "11354", "11368", "11373", "11385", "11432"],
    "BRONX": ["10451", "10453", "10456", "10458", "10462", "10467"],
    "STATEN ISLAND": ["10301", "10304", "10306", "10310", "10312", "10314"],
}

# Realistic mismatched descriptors for common 311 complaint types.
INCONSISTENT_DESCRIPTORS = {
    "Noise - Residential": [
        "Dead Animal on Sidewalk",
        "Pothole in Street",
        "Broken Parking Meter",
    ],
    "Illegal Parking": [
        "No Heat or Hot Water",
        "Loud Music/Party",
        "Rat Sighting",
    ],
    "HEAT/HOT WATER": [
        "Blocked Hydrant",
        "Graffiti on Commercial Building",
        "Abandoned Vehicle",
    ],
    "Street Condition": [
        "Loud Talking",
        "Water Leak in Apartment",
        "Food Poisoning",
    ],
}

DEFAULT_MISMATCH_DESCRIPTORS = [
    "Radioactive Material Spill",
    "Bridge Collapse Hazard",
    "Livestock Loose on Highway",
]


def _empty_gt() -> pd.DataFrame:
    """Returns an empty ground-truth DataFrame with the standard schema."""
    return pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)


def _sample_indices(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Selects a deterministic subset of DataFrame index labels based on rate."""
    if len(df) == 0 or rate <= 0.0:
        return np.array([], dtype=df.index.dtype)
    n_select = max(1, int(round(len(df) * min(rate, 1.0))))
    return rng.choice(df.index.to_numpy(), size=n_select, replace=False)


# ---------------------------------------------------------------------------
# 1. Renamed or dropped column
# ---------------------------------------------------------------------------
def inject_renamed_dropped_column(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates an upstream schema drift by renaming 'incident_zip' -> 'zip_code'
    or dropping 'resolution_description'. Since schema drift is a batch-level
    event, row_id is logged as -1 (or all rows can be considered affected).
    """
    out = df.copy()
    rng = np.random.default_rng(seed)

    # Pick whether to rename a critical column or drop an expected one
    action = rng.choice(["rename", "drop"])
    records = []

    if action == "rename" and "incident_zip" in out.columns:
        out = out.rename(columns={"incident_zip": "zip_code"})
        records.append({
            "row_id": -1,
            "unique_key": "SCHEMA_LEVEL",
            "problem_type": "renamed_dropped_column",
            "column_touched": "incident_zip",
            "original_value": "column:incident_zip",
            "corrupted_value": "renamed:zip_code",
        })
    elif "resolution_description" in out.columns:
        out = out.drop(columns=["resolution_description"])
        records.append({
            "row_id": -1,
            "unique_key": "SCHEMA_LEVEL",
            "problem_type": "renamed_dropped_column",
            "column_touched": "resolution_description",
            "original_value": "column:resolution_description",
            "corrupted_value": "dropped",
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 2. Column type change
# ---------------------------------------------------------------------------
def inject_type_change(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corrupts numeric/formatted columns (incident_zip or latitude) into malformed
    string representations (e.g. float-cast strings like '10027.0' or text words).
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    target_col = "incident_zip" if "incident_zip" in out.columns else "latitude"
    out[target_col] = out[target_col].astype(object)

    chosen_idx = _sample_indices(out, rate, rng)
    records = []

    corrupt_formats = [
        lambda v: f"{v}.0" if pd.notna(v) else "00000.0",
        lambda v: "ten thousand twenty seven",
        lambda v: f"ZIP#{v}",
        lambda v: "N/A - Unknown",
    ]

    for idx in chosen_idx:
        orig = out.at[idx, target_col]
        fmt_fn = rng.choice(corrupt_formats)
        corrupted = fmt_fn(orig)
        out.at[idx, target_col] = corrupted
        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "type_change",
            "column_touched": target_col,
            "original_value": str(orig),
            "corrupted_value": str(corrupted),
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 3. Date format switch mid-batch
# ---------------------------------------------------------------------------
def inject_date_format_switch(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flips created_date from standard ISO (YYYY-MM-DD HH:MM:SS) into ambiguous
    DD/MM/YYYY or verbose text formats mid-batch, simulating a timezone/locale
    serialization bug in an upstream ingestion worker.
    """
    out = df.copy()
    if "created_date" not in out.columns:
        return out, _empty_gt()

    out["created_date"] = out["created_date"].astype(object)
    rng = np.random.default_rng(seed)
    chosen_idx = _sample_indices(out, rate, rng)
    records = []

    for idx in chosen_idx:
        orig = out.at[idx, "created_date"]
        ts = pd.to_datetime(orig, errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp("2023-06-15 14:30:00")

        # Use DD-MM-YYYY (with day > 12 or explicit slashes) or UNIX epoch string
        # so strict ISO parsers fail or misparse.
        style = rng.choice(["dd_mm_yyyy_slash", "verbose_text", "epoch_str"])
        if style == "dd_mm_yyyy_slash":
            corrupted = ts.strftime("%d/%m/%Y %I:%M %p")
        elif style == "verbose_text":
            corrupted = ts.strftime("%b %d, %Y at %H:%Mh")
        else:
            corrupted = f"EPOCH:{int(ts.timestamp())}"

        out.at[idx, "created_date"] = corrupted
        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "date_format_switch",
            "column_touched": "created_date",
            "original_value": str(orig),
            "corrupted_value": str(corrupted),
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 4. Latitude/longitude swapped
# ---------------------------------------------------------------------------
def inject_latlong_swap(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Swaps latitude (~40.7) and longitude (~-73.9) for selected rows.
    """
    out = df.copy()
    if "latitude" not in out.columns or "longitude" not in out.columns:
        return out, _empty_gt()

    rng = np.random.default_rng(seed)
    valid_mask = out["latitude"].notna() & out["longitude"].notna()
    valid_indices = out.index[valid_mask].to_numpy()

    if len(valid_indices) == 0 or rate <= 0.0:
        return out, _empty_gt()

    n_select = max(1, int(round(len(out) * min(rate, 1.0))))
    n_select = min(n_select, len(valid_indices))
    chosen_idx = rng.choice(valid_indices, size=n_select, replace=False)

    records = []
    for idx in chosen_idx:
        orig_lat = out.at[idx, "latitude"]
        orig_lon = out.at[idx, "longitude"]
        out.at[idx, "latitude"] = orig_lon
        out.at[idx, "longitude"] = orig_lat
        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "latlong_swap",
            "column_touched": "latitude,longitude",
            "original_value": f"({orig_lat}, {orig_lon})",
            "corrupted_value": f"({orig_lon}, {orig_lat})",
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 5. Exact duplicates
# ---------------------------------------------------------------------------
def inject_exact_duplicate(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clones existing rows verbatim and appends them to the batch, simulating
    at-least-once webhook delivery without deduplication.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    chosen_idx = _sample_indices(out, rate, rng)
    if len(chosen_idx) == 0:
        return out, _empty_gt()

    cloned_rows = out.loc[chosen_idx].copy()
    # Assign fresh DataFrame indices to the appended clones so every row has a
    # distinct index label for row_id scoring, while unique_key stays identical.
    start_idx = (out.index.max() + 1) if pd.api.types.is_integer_dtype(out.index.dtype) else len(out)
    new_indices = list(range(start_idx, start_idx + len(cloned_rows)))
    cloned_rows.index = new_indices

    out = pd.concat([out, cloned_rows], axis=0)

    records = []
    for orig_idx, new_idx in zip(chosen_idx, new_indices):
        ukey = out.at[new_idx, "unique_key"] if "unique_key" in out.columns else orig_idx
        records.append({
            "row_id": new_idx,
            "unique_key": ukey,
            "problem_type": "exact_duplicate",
            "column_touched": "ALL",
            "original_value": f"clone_of_row:{orig_idx}",
            "corrupted_value": f"duplicate_row:{new_idx}",
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 6. Near-duplicates
# ---------------------------------------------------------------------------
def inject_near_duplicate(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clones existing rows but assigns a new unique_key, shifts created_date by
    1-4 minutes, and adds a minor typo/suffix to descriptor (simulating a user
    submitting the same 311 complaint twice in a row).
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    chosen_idx = _sample_indices(out, rate, rng)
    if len(chosen_idx) == 0:
        return out, _empty_gt()

    cloned_rows = out.loc[chosen_idx].copy()
    start_idx = (out.index.max() + 1) if pd.api.types.is_integer_dtype(out.index.dtype) else len(out)
    new_indices = list(range(start_idx, start_idx + len(cloned_rows)))
    cloned_rows.index = new_indices

    records = []
    suffixes = [" (reported again)", " - urgent!", " [dup]", " near corner"]

    for orig_idx, new_idx in zip(chosen_idx, new_indices):
        orig_desc = str(cloned_rows.at[new_idx, "descriptor"]) if "descriptor" in cloned_rows.columns else ""
        orig_ts = cloned_rows.at[new_idx, "created_date"] if "created_date" in cloned_rows.columns else ""

        # Assign a distinct unique_key so exact-key deduplication misses it
        if "unique_key" in cloned_rows.columns:
            cloned_rows.at[new_idx, "unique_key"] = f"{cloned_rows.at[new_idx, 'unique_key']}_ND"

        # Shift timestamp by 1 to 4 minutes
        if "created_date" in cloned_rows.columns:
            ts = pd.to_datetime(orig_ts, errors="coerce")
            if pd.notna(ts):
                delta_sec = int(rng.integers(60, 240))
                cloned_rows.at[new_idx, "created_date"] = (
                    ts + pd.Timedelta(seconds=delta_sec)
                ).strftime("%Y-%m-%d %H:%M:%S")

        # Slightly perturb descriptor text
        new_desc = orig_desc + str(rng.choice(suffixes))
        if "descriptor" in cloned_rows.columns:
            cloned_rows.at[new_idx, "descriptor"] = new_desc

        records.append({
            "row_id": new_idx,
            "unique_key": cloned_rows.at[new_idx, "unique_key"] if "unique_key" in cloned_rows.columns else new_idx,
            "problem_type": "near_duplicate",
            "column_touched": "created_date,descriptor",
            "original_value": f"row:{orig_idx}|{orig_ts}|{orig_desc}",
            "corrupted_value": f"row:{new_idx}|{cloned_rows.at[new_idx, 'created_date']}|{new_desc}",
        })

    out = pd.concat([out, cloned_rows], axis=0)
    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 7. Null spike in one agency's records
# ---------------------------------------------------------------------------
def inject_null_spike(
    df: pd.DataFrame, rate: float = 0.50, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Picks one well-represented agency in the batch and nulls out key fields
    (descriptor, incident_zip, resolution_description) at a high rate (default
    50% of that agency's rows, or scaled from `rate`), simulating a broken API
    integration for a single city department.
    """
    out = df.copy()
    if "agency" not in out.columns or len(out) == 0:
        return out, _empty_gt()

    rng = np.random.default_rng(seed)
    # Pick among the top 3 largest agencies so the spike is statistically meaningful
    top_agencies = out["agency"].value_counts().head(3).index.tolist()
    if not top_agencies:
        return out, _empty_gt()

    target_agency = rng.choice(top_agencies)
    agency_indices = out.index[out["agency"] == target_agency].to_numpy()

    # Ensure a noticeable spike within that agency (at least 40% of its rows, or `rate` if higher)
    effective_rate = max(rate, 0.45)
    n_select = max(1, int(round(len(agency_indices) * min(effective_rate, 1.0))))
    chosen_idx = rng.choice(agency_indices, size=n_select, replace=False)

    nullable_cols = [
        c for c in ["descriptor", "incident_zip", "resolution_description"] if c in out.columns
    ]
    records = []

    for idx in chosen_idx:
        orig_vals = {c: out.at[idx, c] for c in nullable_cols}
        for c in nullable_cols:
            out.at[idx, c] = np.nan
        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "null_spike",
            "column_touched": ",".join(nullable_cols),
            "original_value": f"agency={target_agency}:{orig_vals}",
            "corrupted_value": "NULL",
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 8. Closed date earlier than created date
# ---------------------------------------------------------------------------
def inject_closed_before_created(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sets closed_date to 1-48 hours BEFORE created_date for selected rows.
    """
    out = df.copy()
    if "created_date" not in out.columns or "closed_date" not in out.columns:
        return out, _empty_gt()

    out["closed_date"] = out["closed_date"].astype(object)
    rng = np.random.default_rng(seed)
    chosen_idx = _sample_indices(out, rate, rng)
    records = []

    for idx in chosen_idx:
        orig_closed = out.at[idx, "closed_date"]
        created_ts = pd.to_datetime(out.at[idx, "created_date"], errors="coerce")
        if pd.isna(created_ts):
            created_ts = pd.Timestamp("2023-05-01 12:00:00")

        hours_back = int(rng.integers(2, 48))
        corrupted_closed = (created_ts - pd.Timedelta(hours=hours_back)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        out.at[idx, "closed_date"] = corrupted_closed
        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "closed_before_created",
            "column_touched": "closed_date",
            "original_value": str(orig_closed),
            "corrupted_value": str(corrupted_closed),
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 9. ZIP code inconsistent with borough
# ---------------------------------------------------------------------------
def inject_zip_borough_mismatch(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replaces incident_zip with a valid NYC ZIP code that belongs to a DIFFERENT
    borough than the row's 'borough' column.
    """
    out = df.copy()
    if "incident_zip" not in out.columns or "borough" not in out.columns:
        return out, _empty_gt()

    out["incident_zip"] = out["incident_zip"].astype(object)
    rng = np.random.default_rng(seed)

    valid_mask = out["borough"].astype(str).str.upper().isin(NYC_BOROUGH_ZIPS.keys())
    candidate_indices = out.index[valid_mask].to_numpy()
    if len(candidate_indices) == 0 or rate <= 0.0:
        return out, _empty_gt()

    n_select = max(1, int(round(len(out) * min(rate, 1.0))))
    n_select = min(n_select, len(candidate_indices))
    chosen_idx = rng.choice(candidate_indices, size=n_select, replace=False)

    records = []
    all_boroughs = list(NYC_BOROUGH_ZIPS.keys())

    for idx in chosen_idx:
        row_borough = str(out.at[idx, "borough"]).upper()
        orig_zip = out.at[idx, "incident_zip"]
        other_boroughs = [b for b in all_boroughs if b != row_borough]
        wrong_borough = rng.choice(other_boroughs)
        wrong_zip = rng.choice(NYC_BOROUGH_ZIPS[wrong_borough])

        out.at[idx, "incident_zip"] = wrong_zip
        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "zip_borough_mismatch",
            "column_touched": "incident_zip",
            "original_value": f"{orig_zip} ({row_borough})",
            "corrupted_value": f"{wrong_zip} (from {wrong_borough})",
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 10. Complaint Type inconsistent with Descriptor
# ---------------------------------------------------------------------------
def inject_complaint_descriptor_mismatch(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairs a row's complaint_type with a semantically incompatible descriptor
    (e.g. 'Noise - Residential' paired with 'Broken Parking Meter').
    """
    out = df.copy()
    if "complaint_type" not in out.columns or "descriptor" not in out.columns:
        return out, _empty_gt()

    out["descriptor"] = out["descriptor"].astype(object)
    rng = np.random.default_rng(seed)
    chosen_idx = _sample_indices(out, rate, rng)
    records = []

    for idx in chosen_idx:
        ctype = str(out.at[idx, "complaint_type"])
        orig_desc = out.at[idx, "descriptor"]
        pool = INCONSISTENT_DESCRIPTORS.get(ctype, DEFAULT_MISMATCH_DESCRIPTORS)
        corrupted_desc = str(rng.choice(pool))

        out.at[idx, "descriptor"] = corrupted_desc
        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "complaint_descriptor_mismatch",
            "column_touched": "descriptor",
            "original_value": f"{ctype} -> {orig_desc}",
            "corrupted_value": f"{ctype} -> {corrupted_desc}",
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 11. Resolution Description contradicting Status
# ---------------------------------------------------------------------------
def inject_resolution_status_contradiction(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Injects a semantic contradiction between 'status' and 'resolution_description'
    (e.g., status='Open' while resolution says the case was inspected and closed).
    """
    out = df.copy()
    if "status" not in out.columns or "resolution_description" not in out.columns:
        return out, _empty_gt()

    out["status"] = out["status"].astype(object)
    out["resolution_description"] = out["resolution_description"].astype(object)
    rng = np.random.default_rng(seed)
    chosen_idx = _sample_indices(out, rate, rng)
    records = []

    open_contradictions = [
        "The Police Department responded to the complaint and took action to fix the condition. Case closed.",
        "Inspector visited the property on Tuesday, confirmed violation was resolved, and closed the ticket.",
        "Issue was fully resolved and closed by the agency supervisor; no further action required.",
    ]
    closed_contradictions = [
        "Complaint is still awaiting initial assignment to a field inspector; investigation has not started.",
        "Pending review. No inspector has visited the site yet and the issue remains active and unresolved.",
        "Work order is currently open and queued for next month's maintenance crew.",
    ]

    for idx in chosen_idx:
        orig_status = str(out.at[idx, "status"])
        orig_res = out.at[idx, "resolution_description"]

        if orig_status.upper() == "CLOSED":
            corrupted_res = str(rng.choice(closed_contradictions))
        else:
            out.at[idx, "status"] = "Open"
            corrupted_res = str(rng.choice(open_contradictions))

        out.at[idx, "resolution_description"] = corrupted_res
        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "resolution_status_contradiction",
            "column_touched": "status,resolution_description",
            "original_value": f"status={orig_status} | res={orig_res}",
            "corrupted_value": f"status={out.at[idx, 'status']} | res={corrupted_res}",
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# 12. Personal data (PII) inside free-text fields
# ---------------------------------------------------------------------------
def inject_pii_in_freetext(
    df: pd.DataFrame, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Injects both structured PII (phone numbers, emails) AND subtle unstructured
    PII (full names and apartment details embedded naturally in prose) so we can
    measure the gap between regex rules and LLM semantic review.
    """
    out = df.copy()
    target_col = "resolution_description" if "resolution_description" in out.columns else "descriptor"
    out[target_col] = out[target_col].astype(object)

    rng = np.random.default_rng(seed)
    chosen_idx = _sample_indices(out, rate, rng)
    records = []

    # Mix regex-catchable PII (phones/emails) with subtle natural-language PII
    # (full names in context) so rules catch ~50% and LLM catches the rest.
    pii_snippets = [
        " Tenant Maria Gonzalez in Apt 4B called from (917) 555-0148 to report the leak.",
        " Spoke with homeowner David K. Morales who asked us to email updates to d.morales84@gmail.com.",
        " Complainant identified himself as Arthur Pendelton residing on the third floor with his wife Clara Pendelton.",
        " Officer interviewed witness Dr. Priya Nair outside her clinic; she confirmed the noise came from Mr. Marcus Vance.",
    ]

    for idx in chosen_idx:
        orig_text = str(out.at[idx, target_col]) if pd.notna(out.at[idx, target_col]) else "Service request logged."
        snippet = str(rng.choice(pii_snippets))
        corrupted_text = orig_text.rstrip(".") + "." + snippet
        out.at[idx, target_col] = corrupted_text

        records.append({
            "row_id": idx,
            "unique_key": out.at[idx, "unique_key"] if "unique_key" in out.columns else idx,
            "problem_type": "pii_in_freetext",
            "column_touched": target_col,
            "original_value": orig_text,
            "corrupted_value": corrupted_text,
        })

    return out, pd.DataFrame(records, columns=GROUND_TRUTH_COLUMNS)


# ---------------------------------------------------------------------------
# Registry & Top-Level Dispatcher
# ---------------------------------------------------------------------------
INJECTOR_REGISTRY: dict[str, Callable[[pd.DataFrame, float, int], tuple[pd.DataFrame, pd.DataFrame]]] = {
    "renamed_dropped_column": inject_renamed_dropped_column,
    "type_change": inject_type_change,
    "date_format_switch": inject_date_format_switch,
    "latlong_swap": inject_latlong_swap,
    "exact_duplicate": inject_exact_duplicate,
    "near_duplicate": inject_near_duplicate,
    "null_spike": inject_null_spike,
    "closed_before_created": inject_closed_before_created,
    "zip_borough_mismatch": inject_zip_borough_mismatch,
    "complaint_descriptor_mismatch": inject_complaint_descriptor_mismatch,
    "resolution_status_contradiction": inject_resolution_status_contradiction,
    "pii_in_freetext": inject_pii_in_freetext,
}


def inject(
    df: pd.DataFrame, problem_type: str, rate: float = 0.05, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-problem entry point matching the assignment stub signature."""
    if problem_type not in INJECTOR_REGISTRY:
        raise ValueError(f"Unknown problem_type '{problem_type}'. Must be one of {PROBLEM_TYPES}")
    return INJECTOR_REGISTRY[problem_type](df, rate, seed)


def inject_batch(
    df: pd.DataFrame,
    problem_types: list[str],
    rate: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies a chosen list of problem types sequentially to a copy of `df` and
    returns `(corrupted_df, combined_ground_truth_df)`.
    """
    corrupted = df.copy()
    gt_logs = []

    for offset, ptype in enumerate(problem_types):
        if ptype not in INJECTOR_REGISTRY:
            raise ValueError(f"Unknown problem_type '{ptype}'.")
        # Derive an independent deterministic seed per problem type
        corrupted, gt = INJECTOR_REGISTRY[ptype](corrupted, rate, seed + offset * 101)
        if not gt.empty:
            gt_logs.append(gt)

    full_gt = pd.concat(gt_logs, ignore_index=True) if gt_logs else _empty_gt()
    return corrupted, full_gt


if __name__ == "__main__":
    # Minimal standalone self-test on a synthetic 20-row NYC 311 batch
    demo_df = pd.DataFrame({
        "unique_key": [f"311-{1000 + i}" for i in range(20)],
        "created_date": ["2023-07-10 09:00:00"] * 20,
        "closed_date": ["2023-07-10 15:00:00"] * 20,
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

    corrupted_demo, gt_demo = inject_batch(demo_df, PROBLEM_TYPES, rate=0.10, seed=42)
    print(f"Original shape: {demo_df.shape} -> Corrupted shape: {corrupted_demo.shape}")
    print("\nGround truth summary by problem type:")
    print(gt_demo["problem_type"].value_counts())