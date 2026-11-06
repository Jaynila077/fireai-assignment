#!/usr/bin/env bash
# run_all.sh: Reproduce all test suites and measurement benchmarks
set -euo pipefail

echo "=== 1. Checking NYC 311 Dataset ==="
if [ ! -f "project_b/data/nyc_311_2023.csv" ]; then
    echo "project_b/data/nyc_311_2023.csv not found; running fetch script..."
    python scripts/fetch_nyc_311_data.py
else
    echo "Found project_b/data/nyc_311_2023.csv"
fi

echo "=== 2. Running All Tests and Benchmarks via Make ==="
make all

echo "=== 3. Verification Complete ==="
echo "Generated benchmark outputs written to:"
echo "  - RESULTS_A_generated.md"
echo "  - RESULTS_B_generated.md"
echo ""
echo "Diff these files against the authoritative RESULTS_A.md / RESULTS_B.md to verify reproducibility."
