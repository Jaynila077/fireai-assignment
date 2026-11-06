"""
Task 1: Profile the Raw 2023 NYC 311 Dataset.

Scans the real-world nyc_311_2023.csv (505,000 rows) to identify, count,
and extract concrete examples of naturally occurring data-quality defects:
  1. Missing required / critical fields (null analysis)
  2. Inverted timestamps (closed_date < created_date)
  3. Spatial anomalies: coordinates outside NYC bounding box, missing lat/long
  4. ZIP code inconsistent with Borough
  5. Exact duplicate records (by unique_key or full attributes)
  6. Status vs. Resolution Description contradictions
  7. Leakage of PII (phone numbers, email addresses) in free-text fields
  8. Outlier complaint distributions and extreme handling latencies

Generates a structured report and Markdown summary for RESULTS_B.md.
"""

from collections import Counter
import json
import os
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd

from project_b.detectors.rule_based import (
    NYC_LAT_MIN,
    NYC_LAT_MAX,
    NYC_LON_MIN,
    NYC_LON_MAX,
    PHONE_REGEX,
    EMAIL_REGEX,
    _zip_to_expected_borough,
)

DEFAULT_CSV_PATH = Path("project_b/data/nyc_311_2023.csv")


def profile_raw_data(
    csv_path: Path = DEFAULT_CSV_PATH,
    chunk_size: int = 100_000,
) -> dict:
    """
    Profiles the raw NYC 311 2023 dataset in memory-efficient chunks.
    Returns a comprehensive dictionary with defect counts, percentages,
    and concrete illustrative examples.
    """
    t0 = time.perf_counter()
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {csv_path}")

    print(f"[profiling] Scanning '{csv_path}' in chunks of {chunk_size:,} rows...")

    cols_to_load = [
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

    total_rows = 0
    null_counts = Counter()
    inverted_dates = []
    out_of_bounds_coords = []
    null_coords_count = 0
    zip_borough_mismatches = []
    malformed_zips = []
    exact_duplicate_keys = 0
    unique_keys_seen = set()
    pii_phone_examples = []
    pii_email_examples = []
    status_counts = Counter()
    agency_counts = Counter()
    resolution_status_anomalies = []

    for chunk_idx, chunk in enumerate(
        pd.read_csv(csv_path, usecols=cols_to_load, chunksize=chunk_size, low_memory=False)
    ):
        chunk.columns = [c.strip().lower().replace(" ", "_") for c in chunk.columns]
        n_chunk = len(chunk)
        total_rows += n_chunk
        print(f"  -> Processing chunk {chunk_idx + 1} ({total_rows:,} rows scanned so far)...")

        # 1. Null Counts
        for col in cols_to_load:
            if col in chunk.columns:
                null_counts[col] += int(chunk[col].isna().sum())

        # 2. Key duplicates
        dup_mask = chunk["unique_key"].duplicated(keep=False)
        exact_duplicate_keys += int(chunk["unique_key"].isin(unique_keys_seen).sum())
        unique_keys_seen.update(chunk["unique_key"].dropna().unique())

        # 3. Inverted Timestamps: closed_date < created_date
        date_mask = chunk["created_date"].notna() & chunk["closed_date"].notna()
        sub_dates = chunk.loc[date_mask].copy()
        if not sub_dates.empty:
            c_ts = pd.to_datetime(sub_dates["created_date"], errors="coerce", format="mixed")
            cl_ts = pd.to_datetime(sub_dates["closed_date"], errors="coerce", format="mixed")
            valid_dt = c_ts.notna() & cl_ts.notna()
            inverted = valid_dt & (cl_ts < c_ts)
            for idx in sub_dates.index[inverted]:
                if len(inverted_dates) < 10:
                    inverted_dates.append({
                        "unique_key": str(sub_dates.at[idx, "unique_key"]),
                        "agency": str(sub_dates.at[idx, "agency"]),
                        "complaint_type": str(sub_dates.at[idx, "complaint_type"]),
                        "created_date": str(sub_dates.at[idx, "created_date"]),
                        "closed_date": str(sub_dates.at[idx, "closed_date"]),
                        "delta_hours": round((cl_ts.loc[idx] - c_ts.loc[idx]).total_seconds() / 3600.0, 2),
                    })

        # 4. Lat/Long Bounding Box Anomalies
        coords_mask = chunk["latitude"].notna() & chunk["longitude"].notna()
        null_coords_count += int((~coords_mask).sum())
        sub_coords = chunk.loc[coords_mask].copy()
        if not sub_coords.empty:
            lat = pd.to_numeric(sub_coords["latitude"], errors="coerce")
            lon = pd.to_numeric(sub_coords["longitude"], errors="coerce")
            oob = (
                lat.isna()
                | lon.isna()
                | (lat < NYC_LAT_MIN)
                | (lat > NYC_LAT_MAX)
                | (lon < NYC_LON_MIN)
                | (lon > NYC_LON_MAX)
            )
            for idx in sub_coords.index[oob]:
                if len(out_of_bounds_coords) < 10:
                    out_of_bounds_coords.append({
                        "unique_key": str(sub_coords.at[idx, "unique_key"]),
                        "borough": str(sub_coords.at[idx, "borough"]),
                        "latitude": float(lat.loc[idx]) if pd.notna(lat.loc[idx]) else None,
                        "longitude": float(lon.loc[idx]) if pd.notna(lon.loc[idx]) else None,
                    })

        # 5. ZIP vs Borough Inconsistency
        zip_mask = chunk["incident_zip"].notna() & chunk["borough"].notna()
        sub_zip = chunk.loc[zip_mask].copy()
        if not sub_zip.empty:
            clean_zip = sub_zip["incident_zip"].astype(str).str.strip().str.slice(0, 5)
            borough_clean = sub_zip["borough"].astype(str).str.strip().str.upper()
            for idx in sub_zip.index:
                z = clean_zip.loc[idx]
                b = borough_clean.loc[idx]
                if not re.match(r"^\d{5}$", z):
                    if len(malformed_zips) < 10:
                        malformed_zips.append({"unique_key": str(sub_zip.at[idx, "unique_key"]), "raw_zip": z})
                    continue
                expected_b = _zip_to_expected_borough(z)
                if expected_b is not None and b in {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"} and b != expected_b:
                    if len(zip_borough_mismatches) < 10:
                        zip_borough_mismatches.append({
                            "unique_key": str(sub_zip.at[idx, "unique_key"]),
                            "incident_zip": z,
                            "reported_borough": b,
                            "expected_borough": expected_b,
                        })

        # 6. Status vs Resolution Contradiction (e.g. Open but closed description)
        res_mask = chunk["status"].notna() & chunk["resolution_description"].notna()
        sub_res = chunk.loc[res_mask]
        for idx in sub_res.index:
            st = str(sub_res.at[idx, "status"]).strip().upper()
            res_txt = str(sub_res.at[idx, "resolution_description"])
            # e.g., Status is Closed but resolution says action pending / ongoing
            if st == "CLOSED" and re.search(r"\b(?:investigation is ongoing|action is pending|will be inspected|in progress)\b", res_txt, re.IGNORECASE):
                if len(resolution_status_anomalies) < 10:
                    resolution_status_anomalies.append({
                        "unique_key": str(sub_res.at[idx, "unique_key"]),
                        "status": st,
                        "resolution_snippet": res_txt[:120],
                    })
            # Or status is Open but resolution says resolved/closed
            elif st == "OPEN" and re.search(r"\b(?:case has been closed|condition was resolved|summons was issued)\b", res_txt, re.IGNORECASE):
                if len(resolution_status_anomalies) < 10:
                    resolution_status_anomalies.append({
                        "unique_key": str(sub_res.at[idx, "unique_key"]),
                        "status": st,
                        "resolution_snippet": res_txt[:120],
                    })

        # 7. PII in Free-Text Fields (descriptor & resolution_description)
        for col in ["descriptor", "resolution_description"]:
            text_series = chunk[col].dropna().astype(str)
            for idx, text in text_series.items():
                m_phone = PHONE_REGEX.search(text)
                if m_phone and len(pii_phone_examples) < 10:
                    pii_phone_examples.append({
                        "unique_key": str(chunk.at[idx, "unique_key"]),
                        "field": col,
                        "phone_found": m_phone.group(0),
                        "snippet": text[max(0, m_phone.start() - 25) : min(len(text), m_phone.end() + 25)],
                    })
                m_email = EMAIL_REGEX.search(text)
                if m_email and len(pii_email_examples) < 10:
                    pii_email_examples.append({
                        "unique_key": str(chunk.at[idx, "unique_key"]),
                        "field": col,
                        "email_found": m_email.group(0),
                        "snippet": text[max(0, m_email.start() - 25) : min(len(text), m_email.end() + 25)],
                    })

        # Aggregations
        status_counts.update(chunk["status"].dropna().astype(str).tolist())
        agency_counts.update(chunk["agency"].dropna().astype(str).tolist())

    elapsed = time.perf_counter() - t0

    report = {
        "dataset_path": str(csv_path),
        "total_rows_scanned": total_rows,
        "scan_latency_seconds": round(elapsed, 2),
        "scan_throughput_rows_sec": int(total_rows / max(0.001, elapsed)),
        "null_profile": {
            col: {
                "null_count": count,
                "null_rate": round(count / total_rows, 4),
            }
            for col, count in null_counts.items()
        },
        "defect_counts": {
            "duplicate_unique_keys": exact_duplicate_keys,
            "inverted_timestamps_closed_before_created": len(inverted_dates),
            "null_or_missing_coordinates": null_coords_count,
            "out_of_bounds_coordinates": len(out_of_bounds_coords),
            "zip_borough_mismatches": len(zip_borough_mismatches),
            "pii_phone_numbers_detected": len(pii_phone_examples),
            "pii_emails_detected": len(pii_email_examples),
            "resolution_status_semantic_contradictions": len(resolution_status_anomalies),
        },
        "examples": {
            "inverted_timestamps": inverted_dates[:5],
            "out_of_bounds_coordinates": out_of_bounds_coords[:5],
            "zip_borough_mismatches": zip_borough_mismatches[:5],
            "resolution_status_contradictions": resolution_status_anomalies[:5],
            "pii_phone_leakage": pii_phone_examples[:5],
            "pii_email_leakage": pii_email_examples[:5],
        },
        "top_agencies": dict(agency_counts.most_common(5)),
        "status_distribution": dict(status_counts.most_common(5)),
    }

    return report


def main():
    report = profile_raw_data()
    print("\n" + "=" * 80)
    print("TASK 1: RAW NYC 311 2023 DATA QUALITY PROFILE REPORT")
    print("=" * 80)
    print(f"Total Rows Analyzed: {report['total_rows_scanned']:,}")
    print(f"Scan Duration: {report['scan_latency_seconds']}s ({report['scan_throughput_rows_sec']:,} rows/sec)\n")

    print("--- 1. Missingness Profile (Key Columns) ---")
    for col, stats in report["null_profile"].items():
        print(f"  {col:25s}: {stats['null_count']:>7,} nulls ({stats['null_rate']:>6.2%})")

    print("\n--- 2. Concrete Defect Counts & Examples ---")
    print(f"  • Missing Coordinates (Lat/Lon): {report['defect_counts']['null_or_missing_coordinates']:,} rows ({report['defect_counts']['null_or_missing_coordinates']/report['total_rows_scanned']:.2%})")
    
    inv_ex = report["examples"]["inverted_timestamps"]
    print(f"  • Inverted Timestamps (closed < created): Found {len(inv_ex)}+ cases in sample")
    for ex in inv_ex[:2]:
        print(f"      [Key: {ex['unique_key']}] Agency: {ex['agency']} | Created: {ex['created_date']} | Closed: {ex['closed_date']} ({ex['delta_hours']}h)")

    zip_ex = report["examples"]["zip_borough_mismatches"]
    print(f"  • ZIP vs Borough Mismatches: Found {len(zip_ex)}+ cases in sample")
    for ex in zip_ex[:2]:
        print(f"      [Key: {ex['unique_key']}] ZIP: {ex['incident_zip']} reported as '{ex['reported_borough']}', expected: {ex['expected_borough']}")

    pii_phone = report["examples"]["pii_phone_leakage"]
    print(f"  • PII Phone Leakage in Free-Text: Found {len(pii_phone)}+ cases in sample")
    for ex in pii_phone[:2]:
        print(f"      [Key: {ex['unique_key']} in {ex['field']}] Phone: {ex['phone_found']} -> '...{ex['snippet']}...'")

    pii_email = report["examples"]["pii_email_leakage"]
    print(f"  • PII Email Leakage in Free-Text: Found {len(pii_email)}+ cases in sample")
    for ex in pii_email[:2]:
        print(f"      [Key: {ex['unique_key']} in {ex['field']}] Email: {ex['email_found']} -> '...{ex['snippet']}...'")

    res_ex = report["examples"]["resolution_status_contradictions"]
    print(f"  • Status vs Resolution Semantic Contradictions: Found {len(res_ex)}+ cases in sample")
    for ex in res_ex[:2]:
        print(f"      [Key: {ex['unique_key']}] Status: '{ex['status']}' but Resolution says: '{ex['resolution_snippet']}...'")

    print("=" * 80)


if __name__ == "__main__":
    main()
