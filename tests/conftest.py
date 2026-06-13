"""Shared fixtures for Econformal integration tests."""

import pytest
import numpy as np
import warnings

# Suppress noisy warnings during test runs
warnings.filterwarnings("ignore")

from econformal.tools.generate_data import generate_test_panel_data


# ── Standard data for DID / SDID (multiple treated units) ──────────
@pytest.fixture(scope="module")
def panel_data_did_sdid():
    """Small balanced panel: 30 units, 5 treated, 5 pre + 3 post periods."""
    return generate_test_panel_data(
        n_ids=30,
        n_treated=5,
        start_year=2010,
        pre_periods=5,
        post_periods=3,
        x_num=1,
        seed=42,
    )


# ── Standard data for SC (exactly 1 treated unit) ─────────────────
@pytest.fixture(scope="module")
def panel_data_sc():
    """Small balanced panel: 30 units, 1 treated, 5 pre + 3 post periods."""
    return generate_test_panel_data(
        n_ids=30,
        n_treated=1,
        start_year=2010,
        pre_periods=5,
        post_periods=3,
        x_num=1,
        seed=42,
    )
