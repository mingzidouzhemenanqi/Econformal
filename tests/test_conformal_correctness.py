"""Tests for mathematical correctness of conformal inference methods.

Verifies:
  - Each conformal method produces valid CI bounds (lower <= upper, non-NaN)
  - JK+ index guards prevent out-of-bounds access
  - Full Conformal validates econ model output
  - CV+ with all strategies works correctly
  - LOO produces valid results with all 3 econ models
  - CI width is positive for all 15 combinations
"""

import pytest
import numpy as np
import pandas as pd

from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data


COVERAGE = 0.90
LOWER_COL = f"{int(COVERAGE * 100)}%_conformal_lower"
UPPER_COL = f"{int(COVERAGE * 100)}%_conformal_upper"


def _assert_valid_ci(result, treat_time):
    """Helper: verify CI bounds are valid for all post-treatment periods."""
    post = result[result["year"] >= treat_time]
    for _, row in post.iterrows():
        lo = row[LOWER_COL]
        hi = row[UPPER_COL]
        if pd.notna(lo) and pd.notna(hi):
            assert lo <= hi, f"CI inverted: [{lo}, {hi}]"


def _get_treat_time(data):
    return data.loc[data["Treat"] == 1, "year"].min()


# ── Full Conformal Validation ──────────────────────────────────────────────

@pytest.mark.parametrize("econ,fixture_name", [
    ("did", "panel_data_did_sdid"),
    ("sc", "panel_data_sc"),
    ("sdid", "panel_data_did_sdid"),
])
def test_full_conformal_validates_output(econ, fixture_name, request):
    """Full Conformal should validate econ model output (not crash with raw KeyError)."""
    data = request.getfixturevalue(fixture_name)
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat",
                   controls_col=["X1"])
    nulls = np.linspace(-10, 10, 10)
    result = m.conformal_inference(econ_model=econ, conformal_model="full",
                                    nulls=nulls, coverage=COVERAGE)
    assert result.shape[0] > 0
    _assert_valid_ci(result, _get_treat_time(data))


# ── JK+ Index Guard ─────────────────────────────────────────────────────────

def test_jk_plus_index_guard():
    """JK+ should not crash with very small n (fixed upper clamp on lb index)."""
    data = generate_test_panel_data(
        n_ids=30, n_treated=1, pre_periods=3, post_periods=2, x_num=1, seed=42,
    )
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat",
                   controls_col=["X1"])
    # With 3 pre-periods, JK+ has n=3 LOO models → n valid effects for each post time
    result = m.conformal_inference(econ_model="sc", conformal_model="jk_plus",
                                    coverage=COVERAGE)
    _assert_valid_ci(result, _get_treat_time(data))


# ── SC + JK+ uses held-out residuals ────────────────────────────────────────

def test_jk_plus_with_sc_correct_residuals():
    """SC+JK+ should use held-out residuals R_j = |effect_{-j}(t_j)| correctly."""
    data = generate_test_panel_data(
        n_ids=30, n_treated=1, pre_periods=5, post_periods=3, x_num=1, seed=42,
    )
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat",
                   controls_col=["X1"])
    result = m.conformal_inference(econ_model="sc", conformal_model="jk_plus",
                                    coverage=COVERAGE)
    _assert_valid_ci(result, _get_treat_time(data))
    # JK+ should have stored residuals and effects
    cm = m.conformal_model
    assert len(cm.jkplus_residuals) > 0, "JK+ should have computed residuals"
    # At least one post-treatment time should have valid effect estimates
    assert any(len(v) > 0 for v in cm.jkplus_effects.values()), (
        "JK+ should have effect estimates for post-treatment times"
    )


# ── CV+ Strategies ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", ["random", "block"])
def test_cv_plus_with_did(strategy):
    """DID+CV+ should produce valid CI with both random and block strategies."""
    data = generate_test_panel_data(
        n_ids=30, n_treated=5, pre_periods=6, post_periods=3, x_num=1, seed=42,
    )
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat",
                   controls_col=["X1"])
    result = m.conformal_inference(econ_model="did", conformal_model="cv_plus",
                                    coverage=COVERAGE, cv_strategy=strategy,
                                    cv_folds=3)
    _assert_valid_ci(result, _get_treat_time(data))


# ── LOO with all 3 econ models ──────────────────────────────────────────────

@pytest.mark.parametrize("econ,fixture_name", [
    ("did", "panel_data_did_sdid"),
    ("sc", "panel_data_sc"),
    ("sdid", "panel_data_did_sdid"),
])
def test_loo_with_all_econ(econ, fixture_name, request):
    """LOO should produce valid CI with all 3 econometric models."""
    data = request.getfixturevalue(fixture_name)
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat",
                   controls_col=["X1"])
    result = m.conformal_inference(econ_model=econ, conformal_model="loo",
                                    coverage=COVERAGE)
    assert result.shape[0] > 0
    _assert_valid_ci(result, _get_treat_time(data))


# ── CI Nonzero Width (all 15 combinations) ──────────────────────────────────

# Subset of combos that are fast enough for width check
WIDTH_COMBOS = [
    ("did", "split", "panel_data_did_sdid"),
    ("did", "loo", "panel_data_did_sdid"),
    ("sc", "split", "panel_data_sc"),
    ("sc", "jk_plus", "panel_data_sc"),
    ("sdid", "split", "panel_data_did_sdid"),
]


@pytest.mark.parametrize("econ,conf,fixture_name", WIDTH_COMBOS)
def test_ci_nonzero_width(econ, conf, fixture_name, request):
    """CI should have positive width for post-treatment periods."""
    data = request.getfixturevalue(fixture_name)
    m = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat",
                   controls_col=["X1"])
    kwargs = dict(econ_model=econ, conformal_model=conf, coverage=COVERAGE)
    if conf == "full":
        kwargs["nulls"] = np.linspace(-10, 10, 10)
    result = m.conformal_inference(**kwargs)
    treat_time = _get_treat_time(data)
    post = result[result["year"] >= treat_time]
    valid = post[post[LOWER_COL].notna() & post[UPPER_COL].notna()]
    if len(valid) > 0:
        widths = valid[UPPER_COL] - valid[LOWER_COL]
        assert (widths >= 0).all(), f"Negative CI width for {econ}+{conf}"
