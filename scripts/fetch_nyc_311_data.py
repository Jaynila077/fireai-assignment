"""
Script to download or extract the NYC 311 Service Requests 2023 extract
for Project B (Data Quality Pipeline).

Dataset: NYC Open Data - 311 Service Requests from 2010 to Present
Resource ID: erm2-nwe9
Socrata API: https://data.cityofnewyork.us/resource/erm2-nwe9.csv

Usage:
    python scripts/fetch_nyc_311_data.py --out project_b/data/nyc_311_2023.csv --limit 505000
"""

import argparse
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request


NYC_311_RESOURCE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"


def fetch_nyc_311(out_path: Path, limit: int = 505_000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 10_000_000:
        print(f"[fetch] '{out_path}' already exists ({out_path.stat().st_size:,} bytes). Skipping download.")
        return

    print(f"[fetch] Fetching up to {limit:,} records from NYC Open Data (2023 extract)...")
    print(f"        Endpoint: {NYC_311_RESOURCE_URL}")

    # Query 2023 calendar year records
    params = {
        "$where": "created_date between '2023-01-01T00:00:00' and '2023-12-31T23:59:59'",
        "$limit": str(limit),
        "$order": "created_date ASC",
    }
    url = f"{NYC_311_RESOURCE_URL}?{urllib.parse.urlencode(params)}"

    print(f"[fetch] Streaming to {out_path}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FireAI-Research-Agent/1.0", "Accept": "text/csv"},
        )
        with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f_out:
            downloaded = 0
            while chunk := resp.read(64 * 1024):
                f_out.write(chunk)
                downloaded += len(chunk)
                if downloaded % (5 * 1024 * 1024) < (64 * 1024):
                    print(f"        Downloaded {downloaded / (1024 * 1024):.1f} MB...")
        print(f"[fetch] Complete! Saved to {out_path} ({out_path.stat().st_size:,} bytes).")
    except Exception as exc:
        print(f"[fetch] Download failed ({exc}).")
        print("        Alternatively, export 2023 NYC 311 data directly from:")
        print("        https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9")
        print(f"        and save as '{out_path}'.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fetch NYC 311 2023 data for Project B.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("project_b/data/nyc_311_2023.csv"),
        help="Target output CSV path (default: project_b/data/nyc_311_2023.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=505_000,
        help="Number of records to fetch (default: 505000)",
    )
    args = parser.parse_args()
    fetch_nyc_311(args.out, args.limit)


if __name__ == "__main__":
    main()
