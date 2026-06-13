"""Final integration tests: all 15 (econ, conformal) combinations.

Comprehensive end-to-end validation of:
  - API correctness (proper invocation, kwargs forwarding)
  - Output validity (shape, columns, CI bounds, no NaN in critical fields)
  - Reproducibility (deterministic with fixed seed)
  - Multi-call safety (same instance, different configs)
"""

import pytest
import numpy as np
import pandas as pd

from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data


COVERAGE = 0.90
NULLS = np.linspace(-10, 10, 15)


def _data(fixture_name, request):
    return request.getfixturevalue(fixture_name)


def _assert_valid(result, treat_time):
    """Full validation of conformal_inference output."""
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] > 0
    lower_col = f"{int(COVERAGE * 100)}%_conformal_lower"
    upper_col = f"{int(COVERAGE * 100)}%_conformal_upper"
    for col in ["effect", lower_col, upper_col]:
        assert col in result.columns, f"Missing {col}"
    assert not result["effect"].isna().any(), "effect has NaN"
    post = result[result["year"] >= treat_time]
    for _, row in post.iterrows():
        lo, hi = row[lower_col], row[upper_col]
        if pd.notna(lo) and pd.notna(hi):
            assert lo <= hi, f"Inverted CI at year={row['year']}"


# ── All 15 combinations ─────────────────────────────────────────────────

ALL_COMBOS = [
    ("did",  "full",    "panel_data_did_sdid", True),
    ("did",  "split",   "panel_data_did_sdid", False),
    ("did",  "loo",     "panel_data_did_sdid", False),
    ("did",  "jk_plus", "panel_data_did_sdid", False),
    ("did",  "cv_plus", "panel_data_did_sdid", False),
    ("sc",   "full",    "panel_data_sc",       True),
    ("sc",   "split",   "panel_data_sc",       False),
    ("sc",   "loo",     "panel_data_sc",       False),
    ("sc",   "jk_plus", "panel_data_sc",       False),
    ("sc",   "cv_plus", "panel_data_sc",       False),
    ("sdid", "full",    "panel_data_did_sdid", True),
    ("sdid", "split",   "panel_data_did_sdid", False),
    ("sdid", "loo",     "panel_data_did_sdid", False),
    ("sdid", "jk_plus", "panel_data_did_sdid", False),
    ("sdid", "cv_plus", "panel_data_did_sdid", False),
]


@pytest.mark.parametrize("econ,conf,fixture_name,needs_nulls", ALL_COMBOS)
def test_all_combinations(econ, conf, fixture_name, needs_nulls, request):
    """Every (econ, conformal) combination returns valid output."""
    data = _data(fixture_name, request)
    treat_time = data.loc[data["Treat"] == 1, "year"].min()
    m = Econformal(data, time="year", id="id", y_col="Y",
                   treat_col="Treat", controls_col=["X1"])
    kw = dict(econ_model=econ, conformal_model=conf, coverage=COVERAGE)
    if needs_nulls:
        kw["nulls"] = NULLS
    result = m.conformal_inference(**kw)
    _assert_valid(result, treat_time)


# ── Reproducibility ──────────────────────────────────────────────────────

def test_reproducibility():
    """Same seed → identical results across two runs."""
    data1 = generate_test_panel_data(
        n_ids=30, n_treated=5, pre_periods=5, post_periods=3, x_num=1, seed=42)
    data2 = generate_test_panel_data(
        n_ids=30, n_treated=5, pre_periods=5, post_periods=3, x_num=1, seed=42)
    r1 = Econformal(data1, time="year", id="id", y_col="Y",
                    treat_col="Treat").conformal_inference(
        econ_model="did", conformal_model="split", coverage=0.9)
    r2 = Econformal(data2, time="year", id="id", y_col="Y",
                    treat_col="Treat").conformal_inference(
        econ_model="did", conformal_model="split", coverage=0.9)
    pd.testing.assert_frame_equal(r1, r2)


# ── Multi-call safety ────────────────────────────────────────────────────

def test_multi_call_same_instance():
    """Two calls on same instance should both produce valid results."""
    data = generate_test_panel_data(
        n_ids=30, n_treated=5, pre_periods=5, post_periods=3, x_num=1, seed=42)
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat")
    r1 = m.conformal_inference(econ_model="did", conformal_model="split",
                                coverage=0.9)
    r2 = m.conformal_inference(econ_model="sdid", conformal_model="split",
                                coverage=0.9)
    assert r1.shape[0] > 0
    assert r2.shape[0] > 0


# ── Custom kwargs ────────────────────────────────────────────────────────

def test_did_full_with_event_window():
    """DID+Full with explicit event_window via kwargs."""
    data = generate_test_panel_data(
        n_ids=30, n_treated=5, pre_periods=5, post_periods=3, x_num=1, seed=42)
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat",
                   controls_col=["X1"])
    result = m.conformal_inference(
        econ_model="did", conformal_model="full",
        nulls=np.linspace(-10, 10, 10), coverage=0.9,
        event_window=(-3, 2),
    )
    treat_time = data.loc[data["Treat"] == 1, "year"].min()
    _assert_valid(result, treat_time)


def test_sdid_with_zeta():
    """SDID with custom zeta and zeta_time via kwargs."""
    data = generate_test_panel_data(
        n_ids=50, n_treated=5, pre_periods=8, post_periods=3, x_num=1, seed=42)
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat")
    result = m.conformal_inference(
        econ_model="sdid", conformal_model="split",
        coverage=0.9, zeta=0.5, zeta_time=0.01,
    )
    assert result.shape[0] > 0


def test_cv_plus_with_custom_folds():
    """CV+ with custom cv_folds and cv_strategy."""
    data = generate_test_panel_data(
        n_ids=30, n_treated=5, pre_periods=10, post_periods=3, x_num=1, seed=42)
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat",
                   controls_col=["X1"])
    result = m.conformal_inference(
        econ_model="did", conformal_model="cv_plus",
        coverage=0.9, cv_folds=5, cv_strategy="block",
    )
    assert result.shape[0] > 0


# ── No controls ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("econ,fixture_name", [
    ("did",  "panel_data_did_sdid"),
    ("sc",   "panel_data_sc"),
    ("sdid", "panel_data_did_sdid"),
])
def test_no_controls(econ, fixture_name, request):
    """Each econ model works without control variables."""
    data = _data(fixture_name, request)
    m = Econformal(data, time="year", id="id", y_col="Y",
                   treat_col="Treat", controls_col=[])
    result = m.conformal_inference(econ_model=econ, conformal_model="split",
                                    coverage=0.9)
    assert result.shape[0] > 0


# ── Coverage values ──────────────────────────────────────────────────────

@pytest.mark.parametrize("cov", [0.80, 0.90, 0.95, 0.99])
def test_various_coverages(cov):
    """Different coverage levels produce valid results."""
    data = generate_test_panel_data(
        n_ids=30, n_treated=5, pre_periods=5, post_periods=3, x_num=1, seed=42)
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat")
    i_cov = int(cov * 100)
    result = m.conformal_inference(econ_model="did", conformal_model="split",
                                    coverage=cov)
    assert f"{i_cov}%_conformal_lower" in result.columns
    assert f"{i_cov}%_conformal_upper" in result.columns
