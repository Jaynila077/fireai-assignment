"""
Statistical / ML-Lite Detectors (Project B, Task 3b).

This is Tier 2: batch-level statistical distributions, robust outlier scores,
and windowed fuzzy matching. Uses zero LLM calls. Returns findings in the exact
same schema as rule_based.py: [row_id, problem_type, detector, confidence, explanation].
"""

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

# Columns monitored for agency-level missingness spikes.
NULL_SPIKE_COLUMNS = ["descriptor", "incident_zip", "resolution_description"]


def _empty_findings() -> pd.DataFrame:
    """Returns an empty findings DataFrame with the standard schema."""
    return pd.DataFrame(columns=FINDINGS_COLUMNS)


# ---------------------------------------------------------------------------
# 7. Null spike in one agency's records (Statistical Binomial / Z-test)
# ---------------------------------------------------------------------------
def detect_null_spike(
    df: pd.DataFrame,
    baseline_rates: dict[str, float] | None = None,
    z_threshold: float = 3.0,
    min_rate_jump: float = 0.25,
) -> pd.DataFrame:
    """
    Detects an abnormal spike in null values for a specific agency compared to
    either a historical `baseline_rates` dict (mapping agency -> expected row
    null rate) or a leave-one-out baseline computed across all other agencies
    in the current batch.
    """
    if "agency" not in df.columns or len(df) == 0:
        return _empty_findings()

    monitored = [c for c in NULL_SPIKE_COLUMNS if c in df.columns]
    if not monitored:
        return _empty_findings()

    # A row is considered part of a multi-column null drop if all monitored fields are null
    # (or if a majority are null when fewer columns exist).
    row_is_null_drop = df[monitored].isna().all(axis=1)

    records = []
    agency_counts = df["agency"].value_counts()

    for agency, n_agency in agency_counts.items():
        if n_agency < 5:
            # Too few rows in this agency to make a reliable statistical claim
            continue

        agency_mask = df["agency"] == agency
        k_null = int(row_is_null_drop[agency_mask].sum())
        observed_rate = k_null / n_agency

        # Determine expected baseline null rate p0
        if baseline_rates and agency in baseline_rates:
            p0 = max(baseline_rates[agency], 0.01)
        else:
            # Leave-one-agency-out baseline rate from the rest of the batch
            other_mask = ~agency_mask
            n_other = int(other_mask.sum())
            p0 = (
                max(float(row_is_null_drop[other_mask].mean()), 0.01)
                if n_other > 0
                else 0.02
            )

        # Binomial standard error under the null hypothesis: SE = sqrt(p0 * (1 - p0) / n)
        se = np.sqrt(p0 * (1.0 - p0) / n_agency)
        z_score = (observed_rate - p0) / se if se > 0 else 0.0

        if z_score >= z_threshold and (observed_rate - p0) >= min_rate_jump:
            conf = float(min(0.99, 0.70 + 0.05 * min(z_score, 6.0)))
            flagged_indices = df.index[agency_mask & row_is_null_drop]
            for idx in flagged_indices:
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
# 6. Near-duplicates (Blocking + Sliding Time Window + RapidFuzz)
# ---------------------------------------------------------------------------
def detect_near_duplicates(
    df: pd.DataFrame,
    text_threshold: float = 80.0,
    time_window_minutes: float = 10.0,
) -> pd.DataFrame:
    """
    Finds near-duplicate service requests that share the same agency,
    complaint_type, and borough within `time_window_minutes`, and have high
    fuzzy text similarity on `descriptor`.

    Why Blocking + RapidFuzz instead of all-pairs TF-IDF?
    - All-pairs cosine similarity on N=500k rows takes O(N^2) memory/compute.
      Blocking by (agency, complaint_type, borough) + sorting by timestamp
      reduces comparisons to a tiny local window O(N * w).
    - `rapidfuzz.fuzz.token_set_ratio` handles extra suffix tokens (e.g.
      'Loud Music/Party' vs 'Loud Music/Party (reported again)') with 100.0
      similarity because one token set is a strict subset of the other.
    """
    required = {"created_date", "complaint_type", "descriptor"}
    if not required.issubset(df.columns) or len(df) < 2:
        return _empty_findings()

    work = df.copy()
    work["_parsed_ts"] = pd.to_datetime(work["created_date"], errors="coerce", format="mixed")
    work["_orig_pos"] = np.arange(len(work))

    # Exclude exact duplicates (already caught by Tier 1 rules) so we don't
    # false-alarm or double-count verbatim clones as near-duplicates.
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

        # Sort by timestamp, breaking ties by original row order so the later row is flagged
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
                    # Because the group is sorted by timestamp, no subsequent row j
                    # can be within `window_delta` of row i.
                    break

                # Do not flag if both timestamp and descriptor are 100% identical
                # and share no variation, unless coordinates also match closely.
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
                # Require either a slight timestamp difference (>0s) or a perturbed
                # descriptor string so routine distinct tickets at the exact same second
                # aren't misflagged unless they match at the exact same coordinate.
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
# 4. Latitude/longitude outliers (Robust Median/MAD Z-Score per Borough)
# ---------------------------------------------------------------------------
def detect_latlong_outliers(
    df: pd.DataFrame, mad_z_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Complements the hard rule-based bounding box by computing a robust Z-score
    using the Median and Median Absolute Deviation (MAD) per borough.
    Why MAD instead of standard mean/std?
    - Swapped coordinates (-73.9 vs +40.7) shift the sample standard deviation
      so violently that standard Z-scores mask outliers when >=5% of rows are
      corrupted. Median and MAD have a 50% breakdown point.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return _empty_findings()

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid_mask = lat.notna() & lon.notna()
    if valid_mask.sum() < 5:
        return _empty_findings()

    # Compute global fallback median and MAD across the batch
    med_lat, med_lon = lat[valid_mask].median(), lon[valid_mask].median()
    # 0.6745 scales MAD to be equivalent to 1 standard deviation for a normal distribution
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
            "explanation": (
                f"Coordinate ({lat.at[idx]}, {lon.at[idx]}) is a statistical outlier "
                f"(robust MAD Z-score={max_z:.1f})."
            ),
        })

    return pd.DataFrame(records, columns=FINDINGS_COLUMNS) if records else _empty_findings()


# ---------------------------------------------------------------------------
# 10. Complaint Type vs. Descriptor Rarity (Empirical Conditional Probability)
# ---------------------------------------------------------------------------
def detect_complaint_descriptor_rarity(
    df: pd.DataFrame, min_type_count: int = 8, prob_threshold: float = 0.08
) -> pd.DataFrame:
    """
    Statistical detector for `complaint_descriptor_mismatch`: estimates the
    empirical conditional probability P(descriptor | complaint_type) within the
    batch and flags rare pairings when a complaint_type has a strongly dominant
    set of descriptors.
    """
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
# Unified Runner for All Statistical Detectors
# ---------------------------------------------------------------------------
STATISTICAL_DETECTORS = {
    "null_spike": detect_null_spike,
    "near_duplicate": detect_near_duplicates,
    "latlong_swap": detect_latlong_outliers,
    "complaint_descriptor_mismatch": detect_complaint_descriptor_rarity,
}


def run_all_statistical_checks(
    df: pd.DataFrame,
    problem_types: list[str] | None = None,
    baseline_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Runs all (or a requested subset of) statistical detectors on `df` and
    returns a single unified findings DataFrame.
    """
    target_types = problem_types or list(STATISTICAL_DETECTORS.keys())
    results = []

    for ptype in target_types:
        if ptype == "null_spike" and "null_spike" in STATISTICAL_DETECTORS:
            res = detect_null_spike(df, baseline_rates=baseline_rates)
        elif ptype in STATISTICAL_DETECTORS:
            res = STATISTICAL_DETECTORS[ptype](df)
        else:
            continue

        if not res.empty:
            results.append(res)

    return pd.concat(results, ignore_index=True) if results else _empty_findings()


if __name__ == "__main__":
    # Standalone self-test verifying statistical detection on injected corruptions
    from project_b.injector.corruption_injector import inject_batch

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

    test_types = ["null_spike", "near_duplicate", "latlong_swap", "complaint_descriptor_mismatch"]
    corrupted_df, gt_df = inject_batch(demo_df, test_types, rate=0.10, seed=42)
    stat_findings = run_all_statistical_checks(corrupted_df)

    print("Ground truth counts:")
    print(gt_df["problem_type"].value_counts())
    print("\nStatistical detector findings:")
    print(stat_findings["problem_type"].value_counts())