"""Integration tests: all 15 combinations of econometric × conformal methods.

Verifies that every combination:
  1. Runs without crashing (correct API usage).
  2. Returns a non-empty DataFrame with the expected columns.
  3. Produces conformal CI bounds where lower <= upper.
  4. Has no NaN values in the effect column.
"""

import pytest
import numpy as np
import pandas as pd
import warnings

from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data

# ── Helpers ────────────────────────────────────────────────────────────
COVERAGE = 0.90
LOWER_COL = f"{int(COVERAGE * 100)}%_conformal_lower"
UPPER_COL = f"{int(COVERAGE * 100)}%_conformal_upper"
NULLS_FAST = np.linspace(-10, 10, 15)  # 快速测试用稀疏网格


def assert_valid_result(result: pd.DataFrame, treat_time: int):
    """验证 conformal_inference 返回的 DataFrame 结构正确且 CI 有效。"""
    # 1. 返回类型
    assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"
    # 2. 非空
    assert result.shape[0] > 0, "Result is empty"
    # 3. 必需列
    for col in ["effect", LOWER_COL, UPPER_COL]:
        assert col in result.columns, f"Missing column: {col} (cols={list(result.columns)})"
    # 4. effect 列无 NaN
    assert not result["effect"].isna().any(), "effect column contains NaN"
    # 5. 处理后时期 CI 上下界有效
    post = result[result["year"] >= treat_time]
    for _, row in post.iterrows():
        lo = row[LOWER_COL]
        hi = row[UPPER_COL]
        # NaN 的 CI 边界允许（如 Split/LOO 在某些时点未覆盖时），但不应出现 hi < lo
        if pd.notna(lo) and pd.notna(hi):
            assert lo <= hi, f"CI inverted at year={row['year']}: [{lo}, {hi}]"


# ── 15-Combination Parametrized Tests ──────────────────────────────────

# 每种组合的配置: (econ_model, conformal_model, data_fixture_name, needs_nulls)
COMBOS = [
    # DID
    ("did", "full",     "panel_data_did_sdid", True),
    ("did", "split",    "panel_data_did_sdid", False),
    ("did", "loo",      "panel_data_did_sdid", False),
    ("did", "jk_plus",  "panel_data_did_sdid", False),
    ("did", "cv_plus",  "panel_data_did_sdid", False),
    # SC
    ("sc",  "full",     "panel_data_sc",       True),
    ("sc",  "split",    "panel_data_sc",       False),
    ("sc",  "loo",      "panel_data_sc",       False),
    ("sc",  "jk_plus",  "panel_data_sc",       False),
    ("sc",  "cv_plus",  "panel_data_sc",       False),
    # SDID
    ("sdid","full",     "panel_data_did_sdid", True),
    ("sdid","split",    "panel_data_did_sdid", False),
    ("sdid","loo",      "panel_data_did_sdid", False),
    ("sdid","jk_plus",  "panel_data_did_sdid", False),
    ("sdid","cv_plus",  "panel_data_did_sdid", False),
]


def _get_treat_time(data):
    """Extract treatment year from generated test data."""
    return data.loc[data["Treat"] == 1, "year"].min()


@pytest.mark.parametrize("econ,conf,fixture_name,needs_nulls", COMBOS)
def test_combination(econ, conf, fixture_name, needs_nulls, request):
    """测试每种计量模型 + 共形方法组合能正确运行并产生有效结果。"""
    data = request.getfixturevalue(fixture_name)
    treat_time = _get_treat_time(data)

    model = Econformal(
        data=data,
        time="year",
        id="id",
        y_col="Y",
        treat_col="Treat",
        controls_col=["X1"],
    )

    kwargs = dict(
        econ_model=econ,
        conformal_model=conf,
        coverage=COVERAGE,
    )
    if needs_nulls:
        kwargs["nulls"] = NULLS_FAST

    result = model.conformal_inference(**kwargs)
    assert_valid_result(result, treat_time)


# ── No-controls variant (DID + Split / SC + Split) ─────────────────────

@pytest.mark.parametrize("econ,fixture_name", [
    ("did",  "panel_data_did_sdid"),
    ("sc",   "panel_data_sc"),
    ("sdid", "panel_data_did_sdid"),
])
def test_no_controls(econ, fixture_name, request):
    """验证 controls_col=[] 时各模型可正常运行。"""
    data = request.getfixturevalue(fixture_name)
    model = Econformal(
        data=data, time="year", id="id", y_col="Y",
        treat_col="Treat", controls_col=[],
    )
    result = model.conformal_inference(
        econ_model=econ, conformal_model="split",
        coverage=COVERAGE,
    )
    assert_valid_result(result, _get_treat_time(data))


# ── Edge Cases: Invalid Inputs ─────────────────────────────────────────

def test_invalid_econ_model_raises(panel_data_did_sdid):
    """无效的计量模型名应抛出 ValueError。"""
    model = Econformal(panel_data_did_sdid, time="year", id="id",
                       y_col="Y", treat_col="Treat")
    with pytest.raises(ValueError):
        model.conformal_inference(econ_model="nonexistent",
                                  conformal_model="split", coverage=0.9)


def test_invalid_conformal_model_raises(panel_data_did_sdid):
    """无效的共形方法名应抛出 ValueError。"""
    model = Econformal(panel_data_did_sdid, time="year", id="id",
                       y_col="Y", treat_col="Treat")
    with pytest.raises(ValueError):
        model.conformal_inference(econ_model="did",
                                  conformal_model="nonexistent", coverage=0.9)


@pytest.mark.parametrize("bad_coverage", [0.0, 1.0, -0.1, 1.5])
def test_coverage_out_of_range_raises(panel_data_did_sdid, bad_coverage):
    """coverage 超出 (0, 1) 时应抛出 ValueError。"""
    model = Econformal(panel_data_did_sdid, time="year", id="id",
                       y_col="Y", treat_col="Treat")
    with pytest.raises(ValueError):
        model.conformal_inference(econ_model="did", conformal_model="split",
                                  coverage=bad_coverage)


def test_missing_y_col_raises(panel_data_did_sdid):
    """不存在的 y_col 应在初始化时抛出 ValueError。"""
    with pytest.raises(ValueError):
        Econformal(panel_data_did_sdid, time="year", id="id",
                   y_col="NONEXISTENT", treat_col="Treat")


def test_missing_controls_col_raises(panel_data_did_sdid):
    """不存在的 controls_col 应在初始化时抛出 ValueError。"""
    with pytest.raises(ValueError):
        Econformal(panel_data_did_sdid, time="year", id="id",
                   y_col="Y", treat_col="Treat",
                   controls_col=["BAD_COL"])


def test_too_few_pre_periods_split_raises():
    """处理前时点不足时 Split Conformal 应抛出 ValueError。"""
    data = generate_test_panel_data(
        n_ids=20, n_treated=5, pre_periods=1, post_periods=3, x_num=0, seed=1,
    )
    model = Econformal(data, time="year", id="id", y_col="Y", treat_col="Treat")
    with pytest.raises(ValueError):
        model.conformal_inference(econ_model="did", conformal_model="split",
                                  coverage=0.9)


# ── Kwargs Pass-through ────────────────────────────────────────────────

def test_kwargs_event_window_passthrough(panel_data_did_sdid):
    """验证 event_window 参数被正确传递给 DID 模型（内外拟合均使用相同窗口）。"""
    model = Econformal(panel_data_did_sdid, time="year", id="id",
                       y_col="Y", treat_col="Treat")
    # 使用较小的 event_window 确保在 conformal 子采样数据中也有足够的相对时间
    result = model.conformal_inference(
        econ_model="did", conformal_model="split",
        coverage=0.9, event_window=(-2, 2),
    )
    assert_valid_result(result, _get_treat_time(panel_data_did_sdid))


def test_kwargs_zeta_passthrough(panel_data_did_sdid):
    """验证 zeta 参数被正确传递给 SDID 模型。"""
    model = Econformal(panel_data_did_sdid, time="year", id="id",
                       y_col="Y", treat_col="Treat")
    result = model.conformal_inference(
        econ_model="sdid", conformal_model="split",
        coverage=0.9, zeta=0.5,
    )
    assert_valid_result(result, _get_treat_time(panel_data_did_sdid))
    assert hasattr(model.econ_model, "_cached_zeta_")


def test_kwargs_random_state_passthrough(panel_data_did_sdid):
    """验证 random_state 参数被正确传递给 SDID placebo 采样。"""
    model = Econformal(panel_data_did_sdid, time="year", id="id",
                       y_col="Y", treat_col="Treat")
    result = model.conformal_inference(
        econ_model="sdid", conformal_model="split",
        coverage=0.9, random_state=99,
    )
    assert_valid_result(result, _get_treat_time(panel_data_did_sdid))
    assert model.econ_model._random_state == 99


# ── Unrecognized kwargs warning ────────────────────────────────────────

def test_unrecognized_kwargs_warns(panel_data_did_sdid):
    """未识别的 kwargs 应触发 UserWarning（防止拼写错误静默忽略）。"""
    model = Econformal(panel_data_did_sdid, time="year", id="id",
                       y_col="Y", treat_col="Treat")
    with pytest.warns(UserWarning, match="未识别"):
        model.conformal_inference(
            econ_model="did", conformal_model="split",
            coverage=0.9, unknown_param="hello",
        )
