"""
Statistical / ML-Lite Detectors (Project B, Task 3b).

This is Tier 2: batch-level statistical distributions, robust outlier scores,
and windowed fuzzy matching. Covers all 12 problem types (4 strong statistical
detectors + 8 deliberately simple statistical baselines for Task 3 completeness).
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

FINDINGS_COLUMNS = [
    "row_id",
    "problem_type",
    "detector",
    "confidence",
    "explanation",
]

EXPECTED_COLUMNS = {
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
}

NULL_SPIKE_COLUMNS = ["descriptor", "incident_zip", "resolution_description"]
CAP_BIGRAM_REGEX = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")


def _empty_findings() -> pd.DataFrame:
    """Returns an empty findings DataFrame with the standard schema."""
    return pd.DataFrame(columns=FINDINGS_COLUMNS)


# ---------------------------------------------------------------------------
# 1. Renamed or dropped column (Jaccard column-set similarity baseline)
# ---------------------------------------------------------------------------
# Expected: detects that column set shifted via Jaccard score < 1.0, but less direct
# than an explicit schema check — included for completeness per Task 3.
def detect_renamed_dropped_column_jaccard(
    df: pd.DataFrame, jaccard_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Computes Jaccard similarity between observed column names and expected
    columns. Flags batch-level schema drift (row_id = -1) if similarity drops.
    """
    observed = set(df.columns)
    intersection = len(EXPECTED_COLUMNS.intersection(observed))
    union = len(EXPECTED_COLUMNS.union(observed))
    jaccard = intersection / union if union > 0 else 1.0

    if jaccard < jaccard_threshold:
        return pd.DataFrame([{
            "row_id": -1,
            "problem_type": "renamed_dropped_column",
            "detector": "statistical",
            "confidence": round(1.0 - jaccard, 2),
            "explanation": f"Column set Jaccard similarity ({jaccard:.2f}) below threshold ({jaccard_threshold}).",
        }], columns=FINDINGS_COLUMNS)
    return _empty_findings()


# ---------------------------------------------------------------------------
# 2. Column type change (String-length Z-score baseline)
# ---------------------------------------------------------------------------
# Expected: lower recall than regex/type rule (catches long strings like '10027.0',
# misses 5-char non-digits) — included for completeness per Task 3.
def detect_type_change_length_zscore(
    df: pd.DataFrame, z_threshold: float = 2.5
) -> pd.DataFrame:
    """
    Treats `incident_zip` string length as a numeric distribution and flags rows
    whose character length deviates from the batch median/std.
    """
    if "incident_zip" not in df.columns or len(df) == 0:
        return _empty_findings()

    non_null = df["incident_zip"].notna()
    lengths = df.loc[non_null, "incident_zip"].astype(str).str.strip().str.len()
    if len(lengths) < 5:
        return _empty_findings()

    med = float(lengths.median())
    std = max(float(lengths.std()), 0.5)
    z_scores = (lengths - med).abs() / std

    records = [
        {
            "row_id": idx,
            "problem_type": "type_change",
            "detector": "statistical",
            "confidence": 0.65,
            "explanation": f"incident_zip length ({lengths.at[idx]}) is a statistical outlier (z={z_scores.at[idx]:.1f}).",
        }
        for idx in lengths.index[z_scores > z_threshold]
    ]
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 3. Date format switch (Digit-ratio statistical baseline)
# ---------------------------------------------------------------------------
# Expected: misses DD/MM/YYYY switches that have the same digit ratio as ISO dates —
# this is the point, showing why regex/schema rules beat statistical char ratios.
def detect_date_format_char_distribution(
    df: pd.DataFrame, z_threshold: float = 2.5
) -> pd.DataFrame:
    """
    Computes the proportion of numeric digit characters in each `created_date`
    string and flags outliers relative to the batch median digit ratio.
    """
    if "created_date" not in df.columns or len(df) == 0:
        return _empty_findings()

    non_null = df["created_date"].notna()
    s = df.loc[non_null, "created_date"].astype(str)
    if len(s) < 5:
        return _empty_findings()

    digit_ratio = s.str.count(r"\d") / s.str.len().clip(lower=1)
    med = float(digit_ratio.median())
    mad = max(float((digit_ratio - med).abs().median()) / 0.6745, 0.02)
    z_scores = (digit_ratio - med).abs() / mad

    records = [
        {
            "row_id": idx,
            "problem_type": "date_format_switch",
            "detector": "statistical",
            "confidence": 0.60,
            "explanation": f"created_date digit ratio ({digit_ratio.at[idx]:.2f}) deviates from median ({med:.2f}).",
        }
        for idx in s.index[z_scores > z_threshold]
    ]
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 4. Latitude/longitude outliers (Robust Median/MAD Z-Score)
# ---------------------------------------------------------------------------
def detect_latlong_outliers(
    df: pd.DataFrame, mad_z_threshold: float = 5.0
) -> pd.DataFrame:
    """Computes robust Z-scores using Median and MAD to spot coordinate swaps/outliers."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return _empty_findings()

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid_mask = lat.notna() & lon.notna()
    if valid_mask.sum() < 5:
        return _empty_findings()

    med_lat, med_lon = lat[valid_mask].median(), lon[valid_mask].median()
    mad_lat = max(float((lat[valid_mask] - med_lat).abs().median()) / 0.6745, 0.05)
    mad_lon = max(float((lon[valid_mask] - med_lon).abs().median()) / 0.6745, 0.05)

    z_lat = (lat - med_lat).abs() / mad_lat
    z_lon = (lon - med_lon).abs() / mad_lon
    outlier_mask = valid_mask & ((z_lat > mad_z_threshold) | (z_lon > mad_z_threshold))

    records = []
    for idx in df.index[outlier_mask]:
        max_z = max(float(z_lat.at[idx]), float(z_lon.at[idx]))
        records.append({
            "row_id": idx,
            "problem_type": "latlong_swap",
            "detector": "statistical",
            "confidence": round(min(0.99, 0.75 + 0.02 * min(max_z, 10.0)), 2),
            "explanation": f"Coordinate ({lat.at[idx]}, {lon.at[idx]}) is a statistical outlier (MAD Z={max_z:.1f}).",
        })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 5. Exact duplicates (Coarse bucket fingerprint baseline)
# ---------------------------------------------------------------------------
# Expected: high false-positive rate compared to exact rule matching because distinct
# requests in the same hour/ZIP/agency collide — included for completeness per Task 3.
def detect_exact_duplicate_coarse_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive statistical/bucketing baseline for exact_duplicate: builds a coarse
    fingerprint from (agency, complaint_type, incident_zip, date-hour) and flags
    collisions. Over-flags legitimate rows that happen in the same hour and ZIP.
    """
    cols = [c for c in ("agency", "complaint_type", "incident_zip") if c in df.columns]
    if not cols or len(df) < 2:
        return _empty_findings()

    work = df[cols].astype(str).agg("|".join, axis=1)
    if "created_date" in df.columns:
        hour_bucket = df["created_date"].astype(str).str.slice(0, 13)
        work = work + "|" + hour_bucket

    collision_mask = work.duplicated(keep="first")
    records = [
        {
            "row_id": idx,
            "problem_type": "exact_duplicate",
            "detector": "statistical",
            "confidence": 0.55,
            "explanation": f"Coarse bucket collision on (agency, complaint_type, zip, hour): {work.at[idx]}.",
        }
        for idx in df.index[collision_mask]
    ]
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 6. Near-duplicates (Blocking + Sliding Time Window + RapidFuzz)
# ---------------------------------------------------------------------------
def detect_near_duplicates(
    df: pd.DataFrame,
    text_threshold: float = 80.0,
    time_window_minutes: float = 10.0,
) -> pd.DataFrame:
    """Finds near-duplicate service requests via blocking, time windowing, and RapidFuzz."""
    required = {"created_date", "complaint_type", "descriptor"}
    if not required.issubset(df.columns) or len(df) < 2:
        return _empty_findings()

    work = df.copy()
    work["_parsed_ts"] = pd.to_datetime(work["created_date"], errors="coerce", format="mixed")
    work["_orig_pos"] = np.arange(len(work))

    if "unique_key" in work.columns:
        exact_dup_mask = work.duplicated(subset=["unique_key"], keep="first")
    else:
        exact_dup_mask = work.duplicated(keep="first")
    work = work.loc[~exact_dup_mask & work["_parsed_ts"].notna()].copy()

    block_cols = [c for c in ("agency", "complaint_type", "borough") if c in work.columns]
    window_delta = pd.Timedelta(minutes=time_window_minutes)

    records = []
    flagged_rows = set()

    for _, group in work.groupby(block_cols, dropna=False, sort=False):
        if len(group) < 2:
            continue

        group_sorted = group.sort_values(by=["_parsed_ts", "_orig_pos"])
        indices = group_sorted.index.to_list()
        timestamps = group_sorted["_parsed_ts"].to_list()
        descriptors = group_sorted["descriptor"].fillna("").astype(str).to_list()
        latitudes = (
            pd.to_numeric(group_sorted["latitude"], errors="coerce").to_list()
            if "latitude" in group_sorted.columns
            else [np.nan] * len(group_sorted)
        )
        longitudes = (
            pd.to_numeric(group_sorted["longitude"], errors="coerce").to_list()
            if "longitude" in group_sorted.columns
            else [np.nan] * len(group_sorted)
        )

        n_grp = len(indices)
        for i in range(n_grp):
            for j in range(i + 1, n_grp):
                if indices[j] in flagged_rows:
                    continue

                time_diff = timestamps[j] - timestamps[i]
                if time_diff > window_delta:
                    break

                if (
                    pd.notna(latitudes[i])
                    and pd.notna(latitudes[j])
                    and (
                        abs(latitudes[i] - latitudes[j]) > 0.005
                        or abs(longitudes[i] - longitudes[j]) > 0.005
                    )
                ):
                    continue

                sim = fuzz.token_set_ratio(descriptors[i], descriptors[j])
                is_perturbed = (descriptors[i] != descriptors[j]) or (
                    pd.notna(latitudes[i])
                    and pd.notna(latitudes[j])
                    and abs(latitudes[i] - latitudes[j]) < 1e-6
                    and abs(longitudes[i] - longitudes[j]) < 1e-6
                    and time_diff > pd.Timedelta(seconds=0)
                )

                if sim >= text_threshold and is_perturbed:
                    flagged_rows.add(indices[j])
                    records.append({
                        "row_id": indices[j],
                        "problem_type": "near_duplicate",
                        "detector": "statistical",
                        "confidence": round(min(0.98, sim / 100.0), 2),
                        "explanation": (
                            f"Near-duplicate of row {indices[i]} "
                            f"(dt={int(time_diff.total_seconds())}s, text_sim={sim:.0f}%)."
                        ),
                    })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 7. Null spike in one agency's records (Statistical Binomial / Z-test)
# ---------------------------------------------------------------------------
def detect_null_spike(
    df: pd.DataFrame,
    baseline_rates: dict[str, float] | None = None,
    z_threshold: float = 3.0,
    min_rate_jump: float = 0.25,
) -> pd.DataFrame:
    """Detects an abnormal spike in null values for a specific agency via binomial Z-score."""
    if "agency" not in df.columns or len(df) == 0:
        return _empty_findings()

    monitored = [c for c in NULL_SPIKE_COLUMNS if c in df.columns]
    if not monitored:
        return _empty_findings()

    row_is_null_drop = df[monitored].isna().all(axis=1)
    records = []
    agency_counts = df["agency"].value_counts()

    for agency, n_agency in agency_counts.items():
        if n_agency < 5:
            continue

        agency_mask = df["agency"] == agency
        k_null = int(row_is_null_drop[agency_mask].sum())
        observed_rate = k_null / n_agency

        if baseline_rates and agency in baseline_rates:
            p0 = max(baseline_rates[agency], 0.01)
        else:
            other_mask = ~agency_mask
            n_other = int(other_mask.sum())
            p0 = max(float(row_is_null_drop[other_mask].mean()), 0.01) if n_other > 0 else 0.02

        se = np.sqrt(p0 * (1.0 - p0) / n_agency)
        z_score = (observed_rate - p0) / se if se > 0 else 0.0

        if z_score >= z_threshold and (observed_rate - p0) >= min_rate_jump:
            conf = float(min(0.99, 0.70 + 0.05 * min(z_score, 6.0)))
            for idx in df.index[agency_mask & row_is_null_drop]:
                records.append({
                    "row_id": idx,
                    "problem_type": "null_spike",
                    "detector": "statistical",
                    "confidence": round(conf, 2),
                    "explanation": (
                        f"Agency '{agency}' null rate spiked to {observed_rate:.1%} "
                        f"vs baseline {p0:.1%} (z={z_score:.1f})."
                    ),
                })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 8. Closed date earlier than created date (Duration Z-score baseline)
# ---------------------------------------------------------------------------
# Expected: much lower recall than the deterministic `closed < created` rule because
# small negative durations (-2h) don't exceed a 2-sigma Z-score when ticket durations
# vary widely — included for completeness per Task 3.
def detect_closed_before_created_duration_zscore(
    df: pd.DataFrame, z_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Naive statistical baseline for closed_before_created: treats resolution
    duration (closed - created) as a Gaussian variable and flags left-tail
    outliers (duration < mean - z * std).
    """
    if "created_date" not in df.columns or "closed_date" not in df.columns:
        return _empty_findings()

    created = pd.to_datetime(df["created_date"], errors="coerce", format="mixed")
    closed = pd.to_datetime(df["closed_date"], errors="coerce", format="mixed")
    duration_hours = (closed - created).dt.total_seconds() / 3600.0
    valid = duration_hours.dropna()
    if len(valid) < 5:
        return _empty_findings()

    mean_dur = float(valid.mean())
    std_dur = max(float(valid.std()), 1.0)
    z_scores = (valid - mean_dur) / std_dur

    records = [
        {
            "row_id": idx,
            "problem_type": "closed_before_created",
            "detector": "statistical",
            "confidence": 0.60,
            "explanation": f"Resolution duration ({valid.at[idx]:.1f}h) is a left-tail outlier (z={z_scores.at[idx]:.1f}).",
        }
        for idx in valid.index[z_scores < -z_threshold]
    ]
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 9. ZIP code inconsistent with borough (Batch majority-vote baseline)
# ---------------------------------------------------------------------------
# Expected: fails whenever an injected wrong ZIP appears only once or twice in the batch
# (since a singleton ZIP has 100% agreement with itself!) — included per Task 3.
def detect_zip_borough_empirical_majority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infers ZIP -> borough mapping purely from the current batch's majority vote
    per ZIP code, then flags rows that disagree with their ZIP's batch majority.
    Fails on rare/singleton ZIPs in small batches.
    """
    if "incident_zip" not in df.columns or "borough" not in df.columns:
        return _empty_findings()

    valid = df["incident_zip"].notna() & df["borough"].notna()
    sub = df.loc[valid, ["incident_zip", "borough"]].astype(str)
    if sub.empty:
        return _empty_findings()

    # Compute majority borough per ZIP within the batch
    majority_map = (
        sub.groupby("incident_zip")["borough"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    records = []
    for idx in sub.index:
        z = sub.at[idx, "incident_zip"]
        b = sub.at[idx, "borough"]
        maj_b = majority_map.get(z, b)
        if b != maj_b:
            records.append({
                "row_id": idx,
                "problem_type": "zip_borough_mismatch",
                "detector": "statistical",
                "confidence": 0.65,
                "explanation": f"Borough '{b}' disagrees with batch-majority borough '{maj_b}' for ZIP {z}.",
            })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 10. Complaint Type vs. Descriptor Rarity (Empirical Conditional Probability)
# ---------------------------------------------------------------------------
def detect_complaint_descriptor_rarity(
    df: pd.DataFrame, min_type_count: int = 8, prob_threshold: float = 0.08
) -> pd.DataFrame:
    """Estimates empirical P(descriptor | complaint_type) and flags rare pairings."""
    if "complaint_type" not in df.columns or "descriptor" not in df.columns:
        return _empty_findings()

    valid = df["complaint_type"].notna() & df["descriptor"].notna()
    sub = df.loc[valid, ["complaint_type", "descriptor"]]
    if sub.empty:
        return _empty_findings()

    type_counts = sub["complaint_type"].value_counts()
    pair_counts = sub.groupby(["complaint_type", "descriptor"]).size()

    records = []
    for idx in sub.index:
        ctype = sub.at[idx, "complaint_type"]
        desc = sub.at[idx, "descriptor"]
        n_total = int(type_counts.get(ctype, 0))
        if n_total < min_type_count:
            continue

        n_pair = int(pair_counts.get((ctype, desc), 0))
        cond_prob = n_pair / n_total

        if cond_prob <= prob_threshold:
            records.append({
                "row_id": idx,
                "problem_type": "complaint_descriptor_mismatch",
                "detector": "statistical",
                "confidence": round(max(0.55, 1.0 - cond_prob * 5), 2),
                "explanation": (
                    f"Rare (complaint_type, descriptor) pair '{ctype}' -> '{desc}' "
                    f"(empirical P={cond_prob:.1%})."
                ),
            })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 11. Resolution contradicting Status (Text-length outlier baseline)
# ---------------------------------------------------------------------------
# Expected: poor accuracy vs LLM semantic review because text length cannot truly
# measure logical contradiction — included for completeness per Task 3.
def detect_resolution_status_length_outlier(
    df: pd.DataFrame, z_threshold: float = 2.2
) -> pd.DataFrame:
    """
    Naive statistical proxy for resolution_status_contradiction: computes the
    character length of `resolution_description` within each `status` group and
    flags length outliers.
    """
    if "status" not in df.columns or "resolution_description" not in df.columns:
        return _empty_findings()

    valid = df["status"].notna() & df["resolution_description"].notna()
    sub = df.loc[valid, ["status", "resolution_description"]].copy()
    if len(sub) < 5:
        return _empty_findings()

    sub["_len"] = sub["resolution_description"].astype(str).str.len()
    records = []

    for status_val, grp in sub.groupby("status"):
        if len(grp) < 3:
            continue
        med = float(grp["_len"].median())
        mad = max(float((grp["_len"] - med).abs().median()) / 0.6745, 5.0)
        z = (grp["_len"] - med).abs() / mad
        for idx in grp.index[z > z_threshold]:
            records.append({
                "row_id": idx,
                "problem_type": "resolution_status_contradiction",
                "detector": "statistical",
                "confidence": 0.50,
                "explanation": (
                    f"Resolution text length ({grp.at[idx, '_len']}) is an outlier "
                    f"for status='{status_val}' (z={z.at[idx]:.1f})."
                ),
            })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 12. Personal data (PII) in free text (Capitalized-word bigram baseline)
# ---------------------------------------------------------------------------
# Expected: misses phones/emails and false-alarms on capitalized agency terms like
# 'Police Department' — included per Task 3 to show why regex + LLM is superior.
def detect_pii_capitalized_bigrams(
    df: pd.DataFrame, min_extra_bigrams: int = 2
) -> pd.DataFrame:
    """
    Naive statistical heuristic for pii_in_freetext: counts Title-Case two-word
    sequences (e.g. 'Maria Gonzalez', but also 'Police Department') in
    `resolution_description` and flags rows exceeding the batch median count.
    """
    target_col = "resolution_description" if "resolution_description" in df.columns else "descriptor"
    if target_col not in df.columns or len(df) == 0:
        return _empty_findings()

    non_null = df[target_col].notna()
    texts = df.loc[non_null, target_col].astype(str)
    if texts.empty:
        return _empty_findings()

    bigram_counts = texts.apply(lambda t: len(CAP_BIGRAM_REGEX.findall(t)))
    baseline_med = float(bigram_counts.median())

    records = [
        {
            "row_id": idx,
            "problem_type": "pii_in_freetext",
            "detector": "statistical",
            "confidence": 0.55,
            "explanation": (
                f"Row contains {bigram_counts.at[idx]} capitalized bigrams "
                f"vs batch median {baseline_med:.0f}."
            ),
        }
        for idx in texts.index[bigram_counts >= (baseline_med + min_extra_bigrams)]
    ]
    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# Unified Registry covering all 12 problem types
# ---------------------------------------------------------------------------
STATISTICAL_DETECTORS = {
    "renamed_dropped_column": detect_renamed_dropped_column_jaccard,
    "type_change": detect_type_change_length_zscore,
    "date_format_switch": detect_date_format_char_distribution,
    "latlong_swap": detect_latlong_outliers,
    "exact_duplicate": detect_exact_duplicate_coarse_fingerprint,
    "near_duplicate": detect_near_duplicates,
    "null_spike": detect_null_spike,
    "closed_before_created": detect_closed_before_created_duration_zscore,
    "zip_borough_mismatch": detect_zip_borough_empirical_majority,
    "complaint_descriptor_mismatch": detect_complaint_descriptor_rarity,
    "resolution_status_contradiction": detect_resolution_status_length_outlier,
    "pii_in_freetext": detect_pii_capitalized_bigrams,
}


def run_all_statistical_checks(
    df: pd.DataFrame,
    problem_types: list[str] | None = None,
    baseline_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Runs all (or a requested subset of) statistical detectors across all 12 problems."""
    target_types = problem_types or list(STATISTICAL_DETECTORS.keys())
    results = []

    for ptype in target_types:
        if ptype == "null_spike":
            res = detect_null_spike(df, baseline_rates=baseline_rates)
        elif ptype in STATISTICAL_DETECTORS:
            res = STATISTICAL_DETECTORS[ptype](df)
        else:
            continue

        if not res.empty:
            results.append(res)

    return pd.concat(results, ignore_index=True) if results else _empty_findings()


if __name__ == "__main__":
    from project_b.injector.corruption_injector import PROBLEM_TYPES, inject_batch

    demo_df = pd.DataFrame({
        "unique_key": [f"311-{3000 + i}" for i in range(40)],
        "created_date": [f"2023-09-01 12:{i:02d}:00" for i in range(40)],
        "closed_date": [f"2023-09-01 18:{i:02d}:00" for i in range(40)],
        "agency": ["NYPD"] * 24 + ["DOT"] * 16,
        "agency_name": ["New York City Police Department"] * 24 + ["Department of Transportation"] * 16,
        "complaint_type": ["Noise - Residential"] * 24 + ["Street Condition"] * 16,
        "descriptor": ["Loud Music/Party"] * 24 + ["Pothole"] * 16,
        "status": ["Closed"] * 40,
        "resolution_description": ["The Police Department responded and took action."] * 40,
        "incident_zip": ["10027"] * 24 + ["11201"] * 16,
        "borough": ["MANHATTAN"] * 24 + ["BROOKLYN"] * 16,
        "latitude": [40.8116] * 24 + [40.6943] * 16,
        "longitude": [-73.9536] * 24 + [-73.9903] * 16,
    })

    corrupted_df, _ = inject_batch(demo_df, PROBLEM_TYPES, rate=0.10, seed=42)
    stat_findings = run_all_statistical_checks(corrupted_df)
    print(f"Registered statistical detectors ({len(STATISTICAL_DETECTORS)}/12):", list(STATISTICAL_DETECTORS.keys()))
    print("\nStatistical detector findings by problem type:")
    print(stat_findings["problem_type"].value_counts())