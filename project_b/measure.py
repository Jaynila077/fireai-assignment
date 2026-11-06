"""
Measurement Harness for Project B (Task 4).

Evaluates all 36 (problem_type x detector_tier) pairs across:
  1. Detection Rate (Recall on injected ground-truth corruptions)
  2. False-Alarm Rate (False Positive Rate on an uncorrupted clean batch)
  3. Cost per 1,000,000 rows ($0.00 for rule/statistical, measured from token usage for LLM)
  4. Throughput (rows/sec on this machine)
  5. Explanation Clarity (qualitative rating + live sample alert text)

Outputs a single Markdown table ready to paste into RESULTS_B.md.
"""

import argparse
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd

from project_b.injector.corruption_injector import PROBLEM_TYPES, inject
from project_b.detectors.rule_based import (
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LON_MAX,
    NYC_LON_MIN,
    ZIP_REGEX,
    run_all_rule_checks,
)
from project_b.detectors.statistical import run_all_statistical_checks
from project_b.detectors.llm_review import review_batch_with_llm

DEFAULT_CSV_PATH = Path("project_b/data/nyc_311_2023.csv")


def _generate_synthetic_clean_pool(n_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Fallback generator if nyc_311_2023.csv is not present locally. Produces
    clean, realistic NYC 311 rows obeying all schema, coordinate, ZIP, and
    semantic constraints.
    """
    rng = np.random.default_rng(seed)
    profiles = [
        ("NYPD", "New York City Police Department", "Noise - Residential", "Loud Music/Party", "MANHATTAN", "10027", 40.8116, -73.9536),
        ("NYPD", "New York City Police Department", "Illegal Parking", "Blocked Hydrant", "BROOKLYN", "11201", 40.6943, -73.9903),
        ("DOT", "Department of Transportation", "Street Condition", "Pothole", "QUEENS", "11101", 40.7447, -73.9485),
        ("HPD", "Department of Housing Preservation and Development", "HEAT/HOT WATER", "ENTIRE BUILDING", "BRONX", "10451", 40.8200, -73.9200),
        ("DOHMH", "Department of Health and Mental Hygiene", "Rodent", "Rat Sighting", "STATEN ISLAND", "10301", 40.6400, -74.0750),
    ]

    choices = rng.integers(0, len(profiles), size=n_rows)
    base_ts = pd.Timestamp("2023-03-01 08:00:00")

    rows = []
    for i, c_idx in enumerate(choices):
        agency, agency_name, ctype, desc, borough, zcode, lat, lon = profiles[c_idx]
        # Space requests 15 minutes apart so clean rows don't trigger near-duplicate windows
        created = base_ts + pd.Timedelta(minutes=15 * i)
        closed = created + pd.Timedelta(hours=int(rng.integers(2, 12)))
        rows.append({
            "unique_key": f"311-{500000 + i}",
            "created_date": created.strftime("%Y-%m-%d %H:%M:%S"),
            "closed_date": closed.strftime("%Y-%m-%d %H:%M:%S"),
            "agency": agency,
            "agency_name": agency_name,
            "complaint_type": ctype,
            "descriptor": desc,
            "status": "Closed",
            "resolution_description": "The responding city agency inspected and resolved the condition.",
            "incident_zip": zcode,
            "borough": borough,
            "latitude": round(lat + float(rng.uniform(-0.008, 0.008)), 6),
            "longitude": round(lon + float(rng.uniform(-0.008, 0.008)), 6),
        })

    return pd.DataFrame(rows)


def load_clean_sample(
    csv_path: Path, sample_size: int, seed: int = 42
) -> pd.DataFrame:
    """
    Loads real NYC 311 data from `csv_path` and pre-filters out pre-existing
    raw-data quality defects (null required fields, inverted dates, out-of-box
    coordinates, malformed ZIPs) so the evaluation baseline is genuinely clean.
    Falls back to synthetic data if the CSV is not found.
    """
    if not csv_path.exists():
        print(f"[measure] '{csv_path}' not found; generating {sample_size} clean synthetic rows.")
        return _generate_synthetic_clean_pool(n_rows=sample_size, seed=seed)

    # Read a chunk large enough to yield `sample_size` clean rows after filtering
    read_limit = max(sample_size * 4, 25000)
    raw_df = pd.read_csv(csv_path, nrows=read_limit, low_memory=False)
    raw_df.columns = [c.strip().lower().replace(" ", "_") for c in raw_df.columns]

    req_cols = [
        "unique_key", "created_date", "closed_date", "agency", "complaint_type",
        "descriptor", "status", "resolution_description", "incident_zip",
        "borough", "latitude", "longitude",
    ]
    present_req = [c for c in req_cols if c in raw_df.columns]
    df = raw_df.dropna(subset=present_req).copy()

    # Filter out pre-existing inverted timestamps and format anomalies
    c_ts = pd.to_datetime(df["created_date"], errors="coerce", format="mixed")
    cl_ts = pd.to_datetime(df["closed_date"], errors="coerce", format="mixed")
    valid_dates = c_ts.notna() & cl_ts.notna() & (cl_ts >= c_ts)
    df = df.loc[valid_dates].copy()
    df["created_date"] = c_ts.loc[df.index].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["closed_date"] = cl_ts.loc[df.index].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Filter to valid 5-digit ZIPs and valid NYC bounding box coordinates
    df["incident_zip"] = df["incident_zip"].astype(str).str.slice(0, 5)
    valid_zip = df["incident_zip"].str.match(ZIP_REGEX)
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid_coords = (
        lat.notna()
        & lon.notna()
        & (lat >= NYC_LAT_MIN)
        & (lat <= NYC_LAT_MAX)
        & (lon >= NYC_LON_MIN)
        & (lon <= NYC_LON_MAX)
    )
    valid_borough = df["borough"].astype(str).str.upper().isin(
        {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"}
    )

    df = df.loc[valid_zip & valid_coords & valid_borough].copy()
    df = df.drop_duplicates(subset=["unique_key"], keep="first")

    # Ensure zero pre-existing rule violations remain in the clean evaluation pool
    pre_existing = run_all_rule_checks(df)
    if not pre_existing.empty:
        dirty_rids = set(pre_existing["row_id"].tolist())
        df = df.loc[~df.index.isin(dirty_rids)].copy()

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    return df.reset_index(drop=True)


def _score_detector(
    gt_df: pd.DataFrame,
    corrupted_findings: pd.DataFrame,
    clean_findings: pd.DataFrame,
    problem_type: str,
    n_clean_rows: int,
) -> tuple[float, float, str]:
    """
    Computes (detection_rate_recall, false_alarm_rate_fpr, sample_explanation).
    Handles both batch-level schema checks (row_id == -1) and row-level checks.
    """
    c_sub = (
        corrupted_findings.loc[corrupted_findings["problem_type"] == problem_type]
        if not corrupted_findings.empty
        else pd.DataFrame()
    )
    cl_sub = (
        clean_findings.loc[clean_findings["problem_type"] == problem_type]
        if not clean_findings.empty
        else pd.DataFrame()
    )

    sample_expl = (
        str(c_sub["explanation"].iloc[0])
        if not c_sub.empty
        else "No alert triggered"
    )

    if problem_type == "renamed_dropped_column":
        # Batch-level schema check: recall is 1.0 if row_id == -1 was flagged on corrupted batch
        recall = 1.0 if (not c_sub.empty and (-1 in c_sub["row_id"].values)) else 0.0
        fpr = 1.0 if not cl_sub.empty else 0.0
        return recall, fpr, sample_expl

    gt_rids = set(gt_df.loc[gt_df["problem_type"] == problem_type, "row_id"].tolist())
    flagged_corrupted_rids = set(c_sub["row_id"].tolist()) if not c_sub.empty else set()
    flagged_clean_rids = set(cl_sub["row_id"].tolist()) if not cl_sub.empty else set()

    recall = (
        len(gt_rids.intersection(flagged_corrupted_rids)) / len(gt_rids)
        if gt_rids
        else 0.0
    )
    fpr = len(flagged_clean_rids) / max(1, n_clean_rows)
    return recall, fpr, sample_expl


def _clarity_label(tier: str, problem_type: str, recall: float) -> str:
    """Assigns a transparent clarity rating based on whether the tier explains root cause."""
    if recall == 0.0:
        return "Low (Missed)"
    if tier == "rule_based":
        return "High (Exact Clause/Value)"
    if tier == "statistical":
        if problem_type in ("null_spike", "near_duplicate", "latlong_swap", "complaint_descriptor_mismatch"):
            return "High (Quantitative Z/Sim)"
        return "Low (Indirect Proxy Metric)"
    if tier == "llm_review":
        if problem_type in ("complaint_descriptor_mismatch", "resolution_status_contradiction", "pii_in_freetext", "null_spike"):
            return "High (Natural Language)"
        return "Medium (Unverified Guess)"
    return "Medium"


def _build_llm_eval_slice(
    corrupted_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    llm_sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constructs a stratified sub-sample of size `llm_sample_size` for the LLM tier
    so we can accurately measure recall and false-alarm rate without burning
    thousands of Gemini free-tier API calls across 12 problem types.
    Preserves all injected ground-truth rows (up to 30% of the slice) plus random
    uncorrupted rows from the batch.
    """
    if len(corrupted_df) <= llm_sample_size:
        return corrupted_df, gt_df, clean_df

    rng = np.random.default_rng(seed)
    gt_rids = [rid for rid in gt_df["row_id"].unique() if rid in corrupted_df.index]
    max_gt = max(1, min(len(gt_rids), int(llm_sample_size * 0.30)))
    chosen_gt = list(rng.choice(gt_rids, size=max_gt, replace=False)) if gt_rids else []

    remaining_pool = [idx for idx in corrupted_df.index if idx not in set(chosen_gt)]
    n_fill = max(0, llm_sample_size - len(chosen_gt))
    chosen_fill = list(rng.choice(remaining_pool, size=min(n_fill, len(remaining_pool)), replace=False))

    combined_idx = sorted(chosen_gt + chosen_fill)
    c_slice = corrupted_df.loc[combined_idx]
    gt_slice = gt_df.loc[
        (gt_df["row_id"].isin(combined_idx)) | (gt_df["row_id"] == -1)
    ].copy()
    clean_slice = clean_df.sample(n=min(llm_sample_size, len(clean_df)), random_state=seed)
    return c_slice, gt_slice, clean_slice


def run_measurements(
    sample_size: int = 2000,
    rate: float = 0.05,
    llm_sample_size: int = 60,
    skip_llm: bool = False,
    seed: int = 42,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """
    Runs the full 12 x 3 evaluation matrix and returns a DataFrame of metrics.
    """
    clean_df = load_clean_sample(csv_path, sample_size=sample_size, seed=seed)
    has_api_key = bool(
        os.environ.get("GEMINI_API_KEY")
        and os.environ.get("GEMINI_API_KEY") != "your_key_here"
    )
    run_llm = (not skip_llm) and has_api_key

    rows = []

    for offset, ptype in enumerate(PROBLEM_TYPES):
        prob_seed = seed + offset * 101
        corrupted_df, gt_df = inject(clean_df, ptype, rate=rate, seed=prob_seed)

        # --- Tier 1: Rule-Based ---
        t0 = time.perf_counter()
        r_corr = run_all_rule_checks(corrupted_df, problem_types=[ptype])
        r_elapsed = max(time.perf_counter() - t0, 1e-6)
        r_clean = run_all_rule_checks(clean_df, problem_types=[ptype])
        r_recall, r_fpr, r_expl = _score_detector(gt_df, r_corr, r_clean, ptype, len(clean_df))
        rows.append({
            "problem_type": ptype,
            "detector_tier": "rule_based",
            "detection_rate": f"{r_recall:.1%}",
            "false_alarm_rate": f"{r_fpr:.2%}",
            "cost_per_1m_rows": "$0.00",
            "throughput_rows_sec": f"{int(len(corrupted_df) / r_elapsed):,}",
            "alert_clarity": _clarity_label("rule_based", ptype, r_recall),
            "sample_explanation": r_expl[:80],
        })

        # --- Tier 2: Statistical ---
        t0 = time.perf_counter()
        s_corr = run_all_statistical_checks(corrupted_df, problem_types=[ptype])
        s_elapsed = max(time.perf_counter() - t0, 1e-6)
        s_clean = run_all_statistical_checks(clean_df, problem_types=[ptype])
        s_recall, s_fpr, s_expl = _score_detector(gt_df, s_corr, s_clean, ptype, len(clean_df))
        rows.append({
            "problem_type": ptype,
            "detector_tier": "statistical",
            "detection_rate": f"{s_recall:.1%}",
            "false_alarm_rate": f"{s_fpr:.2%}",
            "cost_per_1m_rows": "$0.00",
            "throughput_rows_sec": f"{int(len(corrupted_df) / s_elapsed):,}",
            "alert_clarity": _clarity_label("statistical", ptype, s_recall),
            "sample_explanation": s_expl[:80],
        })

        # --- Tier 3: LLM Review ---
        if run_llm:
            c_slice, gt_slice, cl_slice = _build_llm_eval_slice(
                corrupted_df, gt_df, clean_df, llm_sample_size=llm_sample_size, seed=prob_seed
            )
            t0 = time.perf_counter()
            l_corr, usage_corr = review_batch_with_llm(
                c_slice, problem_types=[ptype], batch_size=25, sleep_between_calls=0.3
            )
            l_elapsed = max(time.perf_counter() - t0, 1e-6)
            l_clean, _ = review_batch_with_llm(
                cl_slice, problem_types=[ptype], batch_size=25, sleep_between_calls=0.3
            )
            l_recall, l_fpr, l_expl = _score_detector(
                gt_slice, l_corr, l_clean, ptype, len(cl_slice)
            )
            cost_1m = usage_corr.cost_per_million_rows()
            rows.append({
                "problem_type": ptype,
                "detector_tier": "llm_review",
                "detection_rate": f"{l_recall:.1%}",
                "false_alarm_rate": f"{l_fpr:.2%}",
                "cost_per_1m_rows": f"${cost_1m:.2f}",
                "throughput_rows_sec": f"{len(c_slice) / l_elapsed:.1f}",
                "alert_clarity": _clarity_label("llm_review", ptype, l_recall),
                "sample_explanation": l_expl[:80],
            })
        else:
            rows.append({
                "problem_type": ptype,
                "detector_tier": "llm_review",
                "detection_rate": "SKIPPED (No Key)",
                "false_alarm_rate": "SKIPPED",
                "cost_per_1m_rows": "N/A",
                "throughput_rows_sec": "N/A",
                "alert_clarity": "N/A",
                "sample_explanation": "Set GEMINI_API_KEY to measure live LLM tier.",
            })

    return pd.DataFrame(rows)


def format_markdown_table(results_df: pd.DataFrame) -> str:
    """Formats the measurement DataFrame as a GitHub-flavored Markdown table."""
    headers = [
        "Problem Type",
        "Detector Tier",
        "Detection Rate (Recall)",
        "False-Alarm Rate (FPR)",
        "Cost / 1M Rows",
        "Throughput (rows/s)",
        "Alert Clarity",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in results_df.iterrows():
        cells = [
            f"`{row['problem_type']}`",
            f"**{row['detector_tier']}**",
            str(row["detection_rate"]),
            str(row["false_alarm_rate"]),
            str(row["cost_per_1m_rows"]),
            str(row["throughput_rows_sec"]),
            str(row["alert_clarity"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run Project B data-quality detector measurements across all 12 problem types."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to nyc_311_2023.csv (defaults to project_b/data/nyc_311_2023.csv).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of clean rows to sample for rule and statistical evaluation (default: 5000).",
    )
    parser.add_argument(
        "--llm-sample-size",
        type=int,
        default=60,
        help="Stratified sample size per problem type for LLM evaluation to respect free-tier rate limits (default: 60).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Corruption injection rate per problem type (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and injection (default: 42).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip live Gemini API calls and only measure rule_based and statistical tiers.",
    )
    args = parser.parse_args()

    results_df = run_measurements(
        sample_size=args.sample_size,
        rate=args.rate,
        llm_sample_size=args.llm_sample_size,
        skip_llm=args.skip_llm,
        seed=args.seed,
        csv_path=args.csv_path,
    )

    print("# Project B: Detector Family Comparison (Task 4)\n")
    print(
        f"- **Sample size (Rule / Statistical):** {args.sample_size:,} rows per batch\n"
        f"- **Sample size (LLM Review):** {args.llm_sample_size:,} rows per batch (stratified)\n"
        f"- **Injection rate:** {args.rate:.1%}\n"
        f"- **Seed:** {args.seed}\n"
    )
    print(format_markdown_table(results_df))


if __name__ == "__main__":
    main()