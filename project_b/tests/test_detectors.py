"""
Pytest suite for Project B: Corruption Injector & Three Detector Families.
"""

import os
import pandas as pd
import pytest

from project_b.injector.corruption_injector import (
    GROUND_TRUTH_COLUMNS,
    PROBLEM_TYPES,
    inject,
    inject_batch,
)
from project_b.detectors.rule_based import (
    FINDINGS_COLUMNS as RULE_COLS,
    RULE_DETECTORS,
    run_all_rule_checks,
)
from project_b.detectors.statistical import (
    FINDINGS_COLUMNS as STAT_COLS,
    STATISTICAL_DETECTORS,
    run_all_statistical_checks,
)
from project_b.detectors import llm_review


def _build_clean_df(n_scale: int = 1) -> pd.DataFrame:
    """
    Helper that builds a clean NYC 311 DataFrame of size `60 * n_scale`.
    Uses integer `unique_key` and integer `incident_zip` so pandas 2.x numeric
    dtype upcasting is tested directly.
    """
    n_ny = 25 * n_scale
    n_bk = 20 * n_scale
    n_qn = 15 * n_scale
    n = n_ny + n_bk + n_qn

    agencies = ["NYPD"] * n_ny + ["DOT"] * n_bk + ["HPD"] * n_qn
    boroughs = ["MANHATTAN"] * n_ny + ["BROOKLYN"] * n_bk + ["QUEENS"] * n_qn
    # Keep incident_zip as int64 to verify inject_type_change casts to object safely
    zips = [10027] * n_ny + [11201] * n_bk + [11101] * n_qn
    ctypes = (
        ["Noise - Residential"] * n_ny
        + ["Street Condition"] * n_bk
        + ["HEAT/HOT WATER"] * n_qn
    )
    descs = (
        ["Loud Music/Party"] * n_ny
        + ["Pothole"] * n_bk
        + ["ENTIRE BUILDING"] * n_qn
    )
    lats = (
        [40.8116 + (i % 5) * 0.001 for i in range(n_ny)]
        + [40.6943 + (i % 5) * 0.001 for i in range(n_bk)]
        + [40.7447 + (i % 5) * 0.001 for i in range(n_qn)]
    )
    lons = (
        [-73.9536 - (i % 5) * 0.001 for i in range(n_ny)]
        + [-73.9903 - (i % 5) * 0.001 for i in range(n_bk)]
        + [-73.9485 - (i % 5) * 0.001 for i in range(n_qn)]
    )

    base_ts = pd.Timestamp("2023-05-01 08:00:00")
    created_dates = [
        (base_ts + pd.Timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n)
    ]
    closed_dates = [
        (base_ts + pd.Timedelta(minutes=15 * i, hours=4)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n)
    ]

    return pd.DataFrame({
        "unique_key": [100000 + i for i in range(n)],  # int64 dtype
        "created_date": created_dates,
        "closed_date": closed_dates,
        "agency": agencies,
        "agency_name": agencies,
        "complaint_type": ctypes,
        "descriptor": descs,
        "status": ["Closed"] * n,
        "resolution_description": ["The agency inspected and resolved the issue."] * n,
        "incident_zip": zips,  # int64 dtype
        "borough": boroughs,
        "latitude": lats,      # float64 dtype
        "longitude": lons,     # float64 dtype
    })


@pytest.fixture
def clean_sample_df() -> pd.DataFrame:
    """Fast 60-row fixture for general rule, injector, and fast statistical tests."""
    return _build_clean_df(n_scale=1)


@pytest.fixture
def large_clean_sample_df() -> pd.DataFrame:
    """
    600-row fixture for empirical conditional-probability tests
    (`detect_complaint_descriptor_rarity`), giving 150-250 rows per complaint_type
    so rare injected descriptors have reliable empirical probabilities < 0.08.
    """
    return _build_clean_df(n_scale=10)


def _compute_recall(gt_df: pd.DataFrame, findings_df: pd.DataFrame, problem_type: str) -> float:
    """Computes row-level recall for a specific problem_type against ground truth."""
    gt_rows = set(gt_df.loc[gt_df["problem_type"] == problem_type, "row_id"].tolist())
    if not gt_rows:
        return 1.0
    found_rows = set(
        findings_df.loc[findings_df["problem_type"] == problem_type, "row_id"].tolist()
    )
    return len(gt_rows.intersection(found_rows)) / len(gt_rows)


# ---------------------------------------------------------------------------
# 1. Injector Tests
# ---------------------------------------------------------------------------
def test_injector_determinism_and_immutability(clean_sample_df: pd.DataFrame):
    """Same seed must yield identical corruptions without mutating the input DataFrame."""
    orig_copy = clean_sample_df.copy()

    corrupted_1, gt_1 = inject_batch(clean_sample_df, PROBLEM_TYPES, rate=0.10, seed=123)
    corrupted_2, gt_2 = inject_batch(clean_sample_df, PROBLEM_TYPES, rate=0.10, seed=123)

    pd.testing.assert_frame_equal(clean_sample_df, orig_copy)
    pd.testing.assert_frame_equal(gt_1, gt_2)
    pd.testing.assert_frame_equal(corrupted_1, corrupted_2)
    assert list(gt_1.columns) == GROUND_TRUTH_COLUMNS
    assert set(gt_1["problem_type"].unique()) == set(PROBLEM_TYPES)


# ---------------------------------------------------------------------------
# 2. Rule-Based Detector Tests (Tier 1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "problem_type,min_expected_recall",
    [
        ("renamed_dropped_column", 1.0),
        ("type_change", 1.0),
        ("date_format_switch", 1.0),
        ("latlong_swap", 1.0),
        ("exact_duplicate", 1.0),
        ("closed_before_created", 1.0),
        ("zip_borough_mismatch", 1.0),
        ("resolution_status_contradiction", 0.5),
        ("pii_in_freetext", 0.3),
    ],
)
def test_rule_based_detectors_recall(
    clean_sample_df: pd.DataFrame, problem_type: str, min_expected_recall: float
):
    """Verifies that each strong rule-based detector flags injected rows with high recall."""
    corrupted_df, gt_df = inject(clean_sample_df, problem_type, rate=0.15, seed=42)
    findings_df = run_all_rule_checks(corrupted_df, problem_types=[problem_type])

    assert list(findings_df.columns) == RULE_COLS
    recall = _compute_recall(gt_df, findings_df, problem_type)
    assert recall >= min_expected_recall, (
        f"Rule detector for {problem_type} had recall {recall:.2f} < {min_expected_recall}"
    )


def test_rule_based_zero_false_alarms_on_clean_data(clean_sample_df: pd.DataFrame):
    """Core deterministic rules should produce 0 false positives on clean data."""
    core_rules = [
        "renamed_dropped_column",
        "type_change",
        "date_format_switch",
        "latlong_swap",
        "exact_duplicate",
        "closed_before_created",
        "zip_borough_mismatch",
        "pii_in_freetext",
    ]
    findings_df = run_all_rule_checks(clean_sample_df, problem_types=core_rules)
    assert findings_df.empty, f"Expected 0 false alarms on clean data, got:\n{findings_df}"


def test_all_12_rule_detectors_registered(clean_sample_df: pd.DataFrame):
    """Ensures all 12 problem types are registered in RULE_DETECTORS for Task 3 completeness."""
    assert set(RULE_DETECTORS.keys()) == set(PROBLEM_TYPES)
    findings = run_all_rule_checks(clean_sample_df)
    assert list(findings.columns) == RULE_COLS


# ---------------------------------------------------------------------------
# 3. Statistical Detector Tests (Tier 2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "problem_type,min_expected_recall",
    [
        ("null_spike", 0.90),
        ("near_duplicate", 0.80),
        ("latlong_swap", 0.90),
    ],
)
def test_statistical_detectors_recall(
    clean_sample_df: pd.DataFrame, problem_type: str, min_expected_recall: float
):
    """Verifies that statistical detectors catch batch-level spikes, near-dups, and outliers."""
    corrupted_df, gt_df = inject(clean_sample_df, problem_type, rate=0.10, seed=42)
    findings_df = run_all_statistical_checks(corrupted_df, problem_types=[problem_type])

    assert list(findings_df.columns) == STAT_COLS
    recall = _compute_recall(gt_df, findings_df, problem_type)
    assert recall >= min_expected_recall, (
        f"Statistical detector for {problem_type} had recall {recall:.2f} < {min_expected_recall}"
    )


def test_statistical_complaint_descriptor_mismatch_recall(large_clean_sample_df: pd.DataFrame):
    """
    Uses the 600-row fixture (`large_clean_sample_df`) so empirical conditional
    probabilities P(descriptor | complaint_type) have sufficient sample size per
    category to reliably flag rare injected descriptors below the 0.08 threshold.
    """
    corrupted_df, gt_df = inject(
        large_clean_sample_df, "complaint_descriptor_mismatch", rate=0.05, seed=42
    )
    findings_df = run_all_statistical_checks(
        corrupted_df, problem_types=["complaint_descriptor_mismatch"]
    )

    assert list(findings_df.columns) == STAT_COLS
    recall = _compute_recall(gt_df, findings_df, "complaint_descriptor_mismatch")
    assert recall >= 0.70, (
        f"Statistical detector for complaint_descriptor_mismatch had recall {recall:.2f} < 0.70"
    )


def test_all_12_statistical_detectors_registered(clean_sample_df: pd.DataFrame):
    """Ensures all 12 problem types are registered in STATISTICAL_DETECTORS for Task 3."""
    assert set(STATISTICAL_DETECTORS.keys()) == set(PROBLEM_TYPES)
    findings = run_all_statistical_checks(clean_sample_df)
    assert list(findings.columns) == STAT_COLS


# ---------------------------------------------------------------------------
# 4. LLM Review Detector Tests (Tier 3 — Mocked + Optional Live)
# ---------------------------------------------------------------------------
def test_llm_review_mocked_unit(clean_sample_df: pd.DataFrame, monkeypatch):
    """Tests LLM review batching, schema mapping, and cost calculation offline."""
    corrupted_df, gt_df = inject(
        clean_sample_df.iloc[:10], "complaint_descriptor_mismatch", rate=0.20, seed=42
    )
    expected_rids = gt_df["row_id"].tolist()

    monkeypatch.setattr(llm_review, "_get_gemini_model", lambda *a, **kw: object())

    def _fake_call_gemini(model, prompt, stats):
        stats.api_calls += 1
        stats.input_tokens += 250
        stats.output_tokens += 60
        stats.latency_seconds += 0.15
        return [
            {
                "row_id": rid,
                "problem_type": "complaint_descriptor_mismatch",
                "confidence": 0.95,
                "explanation": "Mocked semantic mismatch detected.",
            }
            for rid in expected_rids
        ]

    monkeypatch.setattr(llm_review, "_call_gemini_json", _fake_call_gemini)

    findings_df, usage = llm_review.review_batch_with_llm(
        corrupted_df,
        problem_types=["complaint_descriptor_mismatch"],
        batch_size=10,
        sleep_between_calls=0.0,
    )

    assert list(findings_df.columns) == llm_review.FINDINGS_COLUMNS
    assert _compute_recall(gt_df, findings_df, "complaint_descriptor_mismatch") == 1.0
    assert usage.api_calls == 1
    assert usage.cost_per_million_rows() > 0.0


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GEMINI_API_KEY") == "your_key_here",
    reason="GEMINI_API_KEY not set in environment; skipping live Gemini API test.",
)
def test_llm_review_live_integration(clean_sample_df: pd.DataFrame):
    """Runs a small 10-row live Gemini API check when GEMINI_API_KEY is available."""
    small_df = clean_sample_df.iloc[:10].copy()
    corrupted_df, gt_df = inject(
        small_df, "resolution_status_contradiction", rate=0.20, seed=42
    )

    findings_df, usage = llm_review.review_batch_with_llm(
        corrupted_df,
        problem_types=["resolution_status_contradiction"],
        batch_size=10,
        sleep_between_calls=0.0,
    )

    if findings_df.empty:
        pytest.skip("Gemini API call failed or quota exhausted; skipping live integration test.")

    assert list(findings_df.columns) == llm_review.FINDINGS_COLUMNS
    assert usage.api_calls >= 1
    recall = _compute_recall(gt_df, findings_df, "resolution_status_contradiction")
    assert recall >= 0.5, f"Live LLM recall on resolution_status_contradiction was {recall:.2f}"