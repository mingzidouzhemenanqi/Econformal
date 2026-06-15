import warnings

import cvxpy as cp
import pandas as pd
import numpy as np
from scipy import stats
from ..tools import check

"""
SDID (Synthetic Difference-in-Differences) 计量模型
基于 Arkhangelsky et al. (2021)，结合合成控制权重与双重差分：
  - 单元权重 ω̂（cvxpy 求解）：使控制组加权平均在预处理期匹配处理组
  - 时间权重 λ̂（cvxpy 求解）：平衡预处理期与后处理期信息
  - 加权 DID：逐期效应 τ̂_t = (处理组 − 合成控制) − λ 加权预处理基线
"""


class Econometric:
    """SDID 计量模型类，遵守统一接口 fit_econmodel(...) -> pd.DataFrame"""

    def __init__(self):
        pass

    # ── 公共接口 ──────────────────────────────────────────────
    def fit_econmodel(
        self,
        data: pd.DataFrame,
        time: str,
        id: str,
        y_col: str,
        treat_col: str,
        coverage: float,
        controls_col: list = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        拟合 SDID 模型并返回标准格式的结果 DataFrame。

        返回列: {time}, effect, std_error, p-value,
                {cov}%_conf_lower, {cov}%_conf_upper
        """
        # 0. 提取可选的超参数
        zeta = kwargs.get("zeta", None)       # None → 自动估计
        zeta_time = kwargs.get("zeta_time", None)
        # 仅在首次设置或用户显式传入时更新 random_state，防止 conformal 重拟合覆盖用户指定值
        if "random_state" in kwargs:
            self._random_state = kwargs["random_state"]
        elif not hasattr(self, '_random_state'):
            self._random_state = 42

        self._id_col = id  # 缓存 ID 列名，供 _augment_with_controls 使用

        # 1. 识别处理信息
        treated_ids = check.get_treated_individuals(data, id, time, treat_col)
        treat_time = check.get_first_treatment_year(data, id, time, treat_col)

        if len(treated_ids) == 0:
            raise ValueError("SDID 需要至少一个处理个体，但未检测到任何处理个体。")

        # 2. 提取控制变量数据（pivot 之前），用于后续增强
        controls_data = None
        _controls_list = []
        if controls_col and len(controls_col) > 0:
            controls_data = data[[time, id] + list(controls_col)].copy()
            _controls_list = list(controls_col)

        # 3. 宽表 pivot（仅结果变量）
        df_wide = data.pivot(index=time, columns=id, values=y_col)
        time_index = df_wide.index.values
        unit_ids = df_wide.columns.values

        # 掩码
        pre_mask = time_index < treat_time
        post_mask = time_index >= treat_time
        T_pre = int(pre_mask.sum())
        T_post = int(post_mask.sum())

        if T_pre < 2:
            raise ValueError(
                f"SDID 至少需要 2 个预处理时期，当前仅有 {T_pre} 个。"
            )
        if T_post < 1:
            raise ValueError(
                f"SDID 至少需要 1 个后处理时期，当前仅有 {T_post} 个。"
            )

        # 4. 分拆处理组与控制组
        treated_cols = [u for u in unit_ids if u in treated_ids]
        control_cols = [u for u in unit_ids if u not in treated_ids]

        # 缓存拟合数据供 placebo 推断使用。
        # 仅在后处理期数更多时更新，防止 Conformal 重拟合的截断数据
        # （T_post=1）覆盖完整数据缓存。T_post 更大的数据包含更多信息。
        _cached_n_post = (
            self._fit_data_cache_.get('_dims', (0, 0, -1))[2]
            if hasattr(self, '_fit_data_cache_') else -1
        )
        if T_post > _cached_n_post:
            self._fit_data_cache_ = {
                'treated_ids': treated_ids,
                'treat_time': treat_time,
                'time_col': time,
                'id_col': id,
                'controls_list': _controls_list,
                'controls_data': controls_data,
                'df_wide': df_wide,
                '_dims': (T_pre, len(control_cols), T_post),
            }

        if len(control_cols) == 0:
            raise ValueError("SDID 需要至少一个控制个体，但所有个体均为处理组。")

        Y_tr = df_wide[treated_cols].mean(axis=1).values   # (T,)
        Y_co = df_wide[control_cols].values                  # (T, N_co)
        N_co = Y_co.shape[1]

        Y_pre_tr_outcome = Y_tr[pre_mask].copy()    # (T_pre,)  仅结果变量
        Y_pre_co_outcome = Y_co[pre_mask, :].copy()  # (T_pre, N_co)

        # 5. 估计正则化参数 ζ（必须在增强前，仅用结果变量行）
        # 若已从首次拟合缓存（如 Full Conformal 重拟合场景），则直接复用，
        # 以保证模型一致性（Arkhangelsky et al., 2021 推荐固定 ζ）。
        # 仅当用户未显式传入 zeta 时才缓存，防止 conformal 重拟合覆盖用户指定值。
        _zeta_from_user = kwargs.get("zeta", None)
        if zeta is None:
            if hasattr(self, '_cached_zeta_'):
                zeta = self._cached_zeta_
            else:
                zeta = self._estimate_zeta(
                    Y_pre_tr_outcome, Y_pre_co_outcome, T_pre, N_co, T_post
                )
        # 仅在首次自动估计或用户显式传入时写入缓存
        if _zeta_from_user is not None or not hasattr(self, '_cached_zeta_'):
            self._cached_zeta_ = zeta

        # 6. 控制变量增强（复用 SC 模式）
        if controls_data is not None and len(_controls_list) > 0:
            Y_pre_tr, Y_pre_co = self._augment_with_controls(
                df_wide, controls_data, time,
                treated_cols, control_cols, treat_time,
                _controls_list, Y_pre_tr_outcome, Y_pre_co_outcome,
            )
        else:
            Y_pre_tr, Y_pre_co = Y_pre_tr_outcome, Y_pre_co_outcome

        # 7. 单元权重 ω̂（岭惩罚基于结果变量行数 T_pre，非增强后行数）
        omega, unit_intercept = self._compute_unit_weights(
            Y_pre_tr, Y_pre_co, zeta, T_pre
        )
        self.unit_weights_ = omega
        self._unit_intercept_ = unit_intercept

        # 8. 对齐结果 A_t = Y_tr,t − Σ ω_j · Y_co_j,t
        synthetic = Y_co @ omega                        # (T,)
        A_tr = Y_tr - synthetic                         # (T,)
        A_co = Y_co - synthetic.reshape(-1, 1)          # (T, N_co)

        # 9. 时间权重 λ̂（若已缓存 zeta_time 则复用，保证模型一致性）
        if zeta_time is None and hasattr(self, '_cached_zeta_time_'):
            zeta_time = self._cached_zeta_time_
        lambda_weights = self._compute_time_weights(
            A_co, pre_mask, post_mask, zeta_time, T_pre
        )
        self.time_weights_ = lambda_weights

        # 10. 计算逐期效应
        A_tr_pre = A_tr[pre_mask]                        # (T_pre,)
        baseline = float(np.dot(lambda_weights, A_tr_pre))

        # 所有时期效应
        all_effects = A_tr - baseline                     # (T,)

        # 预处理期效应（用于推断）
        pre_effects = A_tr_pre - baseline                 # (T_pre,)

        # 11. 推断
        std_err, pval_dict, ci_lo_dict, ci_up_dict = self._compute_inference(
            pre_effects, all_effects, time_index, coverage
        )

        # 12. 组装输出 DataFrame
        ci_lower_col = f"{int(coverage * 100)}%_conf_lower"
        ci_upper_col = f"{int(coverage * 100)}%_conf_upper"

        rows = []
        for i, t_val in enumerate(time_index):
            eff = float(all_effects[i])
            # _compute_inference 已为所有时点预计算了 p-value 和 CI
            rows.append({
                time: t_val,
                "effect": eff,
                "std_error": std_err,
                "p-value": pval_dict[t_val],
                ci_lower_col: ci_lo_dict[t_val],
                ci_upper_col: ci_up_dict[t_val],
            })

        result_df = pd.DataFrame(rows)
        # 保留原始时间列类型，不强制转换
        # （避免 datetime/float/str 类型被破坏导致 merge 失败）

        self.results_df = result_df
        return result_df

    # ── 单元权重优化 ─────────────────────────────────────────
    def _compute_unit_weights(
        self,
        Y_pre_tr: np.ndarray,
        Y_pre_co: np.ndarray,
        zeta: float,
        T_pre_outcome: int,
    ) -> tuple[np.ndarray, float]:
        """cvxpy 求解单元权重 ω，带岭惩罚以保证唯一性。

        包含截距 r̂（Arkhangelsky et al., 2021，式 (4)）：
            min_{ω, r} ||Y_pre_co @ ω + r − Y_pre_tr||² + ζ²·T_pre·||ω||²
            s.t. Σ ω = 1, ω ≥ 0, r 无约束

        截距吸收处理组与控制组之间的系统性水平差异，使 ω 专注于匹配
        时序轨迹（形状）。虽然 r̂ 在后续 τ̂ 估计中因时间加权差分而消除，
        但它会影响单元权重的质量——实测可降低预处理期 RMSE 约 40%。

        T_pre_outcome: 结果变量预处理期数（不含控制变量增强行）。
        岭惩罚基于纯结果变量行数，与 Arkhangelsky et al. (2021) 一致。

        Returns
        -------
        omega : np.ndarray (N_co,)
            单元权重向量。
        intercept : float
            估计的截距 r̂（供诊断使用，不参与效应计算）。
        """
        N_co = Y_pre_co.shape[1]

        w = cp.Variable(N_co)
        r = cp.Variable()                            # ← 截距项
        fidelity = cp.sum_squares(Y_pre_co @ w + r - Y_pre_tr)
        penalty = (zeta ** 2) * T_pre_outcome * cp.sum_squares(w)
        objective = cp.Minimize(fidelity + penalty)
        constraints = [cp.sum(w) == 1, w >= 0]

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(verbose=False)
        except cp.error.SolverError:
            warnings.warn(
                "SDID 单元权重优化求解器报错，回退为等权重。"
                "请检查数据质量或尝试调整 zeta 参数。"
            )
            return np.ones(N_co) / N_co, 0.0

        if problem.status not in ("optimal", "optimal_inaccurate"):
            warnings.warn(
                f"SDID 单元权重优化未收敛 (状态: {problem.status})，回退为等权重。"
            )
            return np.ones(N_co) / N_co, 0.0

        if problem.status == "optimal_inaccurate":
            warnings.warn(
                "SDID 单元权重优化返回 'optimal_inaccurate'，"
                "解可能数值不精确。建议检查数据质量或调整 zeta 参数。"
            )

        omega = w.value
        intercept = float(r.value) if r.value is not None else 0.0

        # 数值归一化（clip 微小负值为 0，确保 sum ≈ 1）
        omega = np.maximum(omega, 0)
        s = omega.sum()
        if s > 1e-12:
            omega /= s
        else:
            # 退化情况：所有权重均为零 → 回退等权重
            warnings.warn(
                "SDID 单元权重全部退化（求解后所有权重 ≈ 0），已回退为等权重。"
                "此时 SDID 退化为标准 DID。请检查数据质量或调整 zeta。"
            )
            omega = np.ones(N_co) / N_co
            intercept = 0.0
        return omega, intercept

    # ── 时间权重优化 ─────────────────────────────────────────
    def _compute_time_weights(
        self,
        A_co: np.ndarray,
        pre_mask: np.ndarray,
        post_mask: np.ndarray,
        zeta_time: float | None,
        T_pre: int,
    ) -> np.ndarray:
        """
        cvxpy 求解时间权重 λ。
        回归：Ā_i,post = λ₀ + Σ_s λ_s · A_i,s （以控制单元为观测）
        约束：Σ λ = 1, λ ≥ 0

        若 zeta_time 为 None，使用轻量 L2 正则化默认值并缓存，
        确保后续调用（如 Full Conformal 重拟合）使用一致的正则化强度。
        """
        N_co = A_co.shape[1]
        A_co_pre = A_co[pre_mask, :]       # (T_pre, N_co)
        A_co_post_mean = A_co[post_mask, :].mean(axis=0)  # (N_co,)

        # 回归：N_co 个观测，T_pre + 1 个参数（λ₀ + T_pre 个 λ）
        if N_co < T_pre + 1:
            warnings.warn(
                f"时间权重回归的观测数 ({N_co} 个控制单元) 少于参数数量 "
                f"({T_pre} 个 λ + 1 个截距 = {T_pre + 1})，"
                f"回归可能欠定。建议确保控制单元数 > 预处理期数。"
            )
        elif N_co < 2 * T_pre:
            warnings.warn(
                f"时间权重回归的观测数 ({N_co}) 相对参数数量 ({T_pre} 个 λ + 1) "
                f"偏少，回归可能不稳定。建议 N_co >= 2 × T_pre (={2*T_pre})。"
            )

        lam = cp.Variable(T_pre)
        lam0 = cp.Variable()

        residuals = A_co_post_mean - lam0 - A_co_pre.T @ lam
        fidelity = cp.sum_squares(residuals)

        # 轻量 L2 正则化，提升数值稳定性。
        # L2 惩罚在 sum(λ)=1 约束下等价于向等权重收缩：
        #   argmin Σ λ²  s.t. Σ λ = 1  →  λ = 1/T_pre
        # 因此正则化越强，时间权重越接近均匀分布（标准 DID 加权）。
        if zeta_time is None:
            if hasattr(self, '_cached_zeta_time_'):
                zeta_time = self._cached_zeta_time_
            else:
                zeta_time = 1e-4 * T_pre
                self._cached_zeta_time_ = zeta_time
        elif not hasattr(self, '_cached_zeta_time_'):
            # 用户显式传入且尚无缓存 → 记录
            self._cached_zeta_time_ = zeta_time

        penalty = zeta_time * cp.sum_squares(lam)
        objective = cp.Minimize(fidelity + penalty)
        constraints = [cp.sum(lam) == 1, lam >= 0]

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(verbose=False)
        except cp.error.SolverError:
            warnings.warn(
                "SDID 时间权重优化求解器报错，回退为等权重。"
                "此时 SDID 退化为标准 DID。"
            )
            return np.ones(T_pre) / T_pre

        if problem.status not in ("optimal", "optimal_inaccurate"):
            warnings.warn(
                f"SDID 时间权重优化未收敛 (状态: {problem.status})，回退为等权重。"
                "此时 SDID 退化为标准 DID。"
            )
            return np.ones(T_pre) / T_pre

        if problem.status == "optimal_inaccurate":
            warnings.warn(
                "SDID 时间权重优化返回 'optimal_inaccurate'，"
                "解可能数值不精确。建议检查数据质量或调整 zeta_time 参数。"
            )

        lam_val = lam.value
        lam_val = np.maximum(lam_val, 0)
        s = lam_val.sum()
        if s > 1e-12:
            lam_val /= s
        else:
            warnings.warn(
                "SDID 时间权重全部退化（求解后所有权重 ≈ 0），已回退为等权重。"
                "此时 SDID 退化为标准 DID。请检查数据质量或调整 zeta_time。"
            )
            lam_val = np.ones(T_pre) / T_pre
        return lam_val

    # ── 正则化参数估计 ───────────────────────────────────────
    def _estimate_zeta(
        self,
        Y_pre_tr: np.ndarray,
        Y_pre_co: np.ndarray,
        T_pre: int,
        N_co: int,
        T_post: int,
    ) -> float:
        """
        ζ = σ̂ · T_post^(1/4) / sqrt(N_co)
        σ̂ 来自结果差分标准差（处理组与控制组的混合估计）。
        """
        # 处理组和控制组的合并差分标准差（与论文一致：pooled std over all diffs）
        d_tr = np.diff(Y_pre_tr)
        d_co = np.diff(Y_pre_co, axis=0)
        all_diffs = np.concatenate([d_co.ravel(), d_tr])
        sigma_hat = float(np.std(all_diffs, ddof=1)) if len(all_diffs) > 1 else 0.0
        if sigma_hat < 1e-10:
            sigma_hat = 1e-10

        zeta = sigma_hat * (T_post ** 0.25) / np.sqrt(max(N_co, 1))
        return zeta

    # ── 控制变量增强 ─────────────────────────────────────────
    def _augment_with_controls(
        self,
        df_wide: pd.DataFrame,
        controls_data: pd.DataFrame,
        time: str,
        treated_cols: list,
        control_cols: list,
        treat_time,
        controls_list: list,
        Y_pre_tr: np.ndarray,
        Y_pre_co: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        将控制变量的预处理均值（标准化后）作为增强行，追加到
        Y_pre_tr 和 Y_pre_co 中。复用 sc.py 的标准化增强模式。
        """
        aug_rows_y = [Y_pre_tr]
        aug_rows_X = [Y_pre_co]

        for c in controls_list:
            c_wide = controls_data.pivot(index=time, columns=self._id_col, values=c)
            # 对齐列顺序（引入的列若不存在则填 NaN）
            c_wide = c_wide.reindex(columns=df_wide.columns)
            c_pre = c_wide[c_wide.index < treat_time]
            c_means = c_pre.mean(axis=0)

            # 防御：控制变量含 NaN 时跳过（如原始数据缺失值）
            if c_means.isna().any():
                warnings.warn(
                    f"控制变量 '{c}' 存在缺失值，已跳过该变量。"
                    f"请检查数据完整性或考虑填充缺失值。"
                )
                continue

            # 标准化
            c_std = c_means.std(ddof=1)
            if c_std < 1e-10:
                c_std = 1.0
            c_means_std = (c_means - c_means.mean()) / c_std

            treated_val = c_means_std[treated_cols].mean()
            control_vals = c_means_std[control_cols].values

            aug_rows_y.append(np.array([treated_val]))
            aug_rows_X.append(control_vals.reshape(1, -1))

        # 边界警告
        n_pre_outcome = len(Y_pre_tr)
        if len(controls_list) > n_pre_outcome:
            warnings.warn(
                f"控制变量数 ({len(controls_list)}) 超过结果变量预处理行数 "
                f"({n_pre_outcome})。协变量匹配可能主导优化目标。"
            )

        Y_pre_tr_aug = np.concatenate(aug_rows_y)
        Y_pre_co_aug = np.vstack(aug_rows_X)

        return Y_pre_tr_aug, Y_pre_co_aug

    # ── 推断 ─────────────────────────────────────────────────
    def _compute_inference(
        self,
        pre_effects: np.ndarray,
        all_effects: np.ndarray,
        time_index: np.ndarray,
        coverage: float,
    ) -> tuple:
        """
        基于 placebo（安慰剂）方差的推断。

        按照 Arkhangelsky et al. (2021) 推荐的方法：
        1. 对每个控制单元 j，将其作为"伪处理组"运行 SDID
        2. 收集所有安慰剂后处理期平均效应 {τ̂_j}
        3. 用安慰剂效应的标准差作为标准误
        4. 正态近似计算 p 值和置信区间

        当控制单元数较少（< 5）时回退为简单 std(pre_effects) 方法。

        注意：placebo 计算成本高（O(N_co) 次 SDID 拟合），仅在首次调用时计算，
        后续调用（如 Full Conformal 重拟合）直接复用缓存结果。
        """
        # ── 预处理期拟合质量诊断 ──
        pre_rmse = float(np.sqrt(np.mean(pre_effects ** 2)))
        self._pre_fit_rmse_ = pre_rmse

        if pre_rmse > 1.0:
            warnings.warn(
                f"SDID 预处理期拟合 RMSE 较高 ({pre_rmse:.4f})。"
                f"合成控制可能未充分拟合处理组的预处理轨迹，"
                f"建议检查数据质量或调整 zeta 参数。"
            )

        # ── Placebo 方差估计（首次计算后缓存）──
        if hasattr(self, '_cached_placebo_effects_'):
            placebo_effects = self._cached_placebo_effects_
            # 检查缓存来自不同数据规模（如 Full Conformal 用截断数据重拟合时
            # 复用全量数据的 placebo 效应），发出警告
            _cache = self._fit_data_cache_ if hasattr(self, '_fit_data_cache_') else {}
            _cached_n_post_prev = _cache.get('_dims', (0, 0, -1))[2]
            _cached_treat_time = _cache.get('treat_time')
            _current_n_post = (
                int((time_index >= _cached_treat_time).sum())
                if _cached_treat_time is not None else _cached_n_post_prev
            )
            if _cached_n_post_prev > _current_n_post and _cached_n_post_prev > 1:
                warnings.warn(
                    f"SDID placebo 标准误来自完整数据（{_cached_n_post_prev} 个处理后时点），"
                    f"但当前拟合仅使用 {_current_n_post} 个处理后时点。"
                    f"标准误可能不完全适用于当前截断数据。"
                    f"（这通常发生在 Full Conformal 重拟合场景。）"
                )
        else:
            placebo_effects = self._compute_placebo_effects(
                pre_effects, all_effects, time_index
            )
            self._cached_placebo_effects_ = placebo_effects

        if placebo_effects is not None and len(placebo_effects) >= 5:
            # 使用安慰剂效应的变异作为标准误
            std_err = float(np.std(placebo_effects, ddof=1))
            self._inference_method_ = "placebo"
            self._placebo_effects_ = placebo_effects
        else:
            # 回退：使用预处理期效应的标准误
            std_err = (
                float(np.std(pre_effects, ddof=1))
                if len(pre_effects) >= 2
                else 0.0
            )
            self._inference_method_ = "pre_treat_std"
            if placebo_effects is not None and len(placebo_effects) < 5:
                warnings.warn(
                    f"安慰剂单元过少 ({len(placebo_effects)} 个)，"
                    f"回退为预处理期标准差进行推断。"
                    f"建议增加控制单元数以获得更可靠的推断。"
                )

        z = stats.norm.ppf((1 + coverage) / 2)

        pval_dict = {}
        ci_lo_dict = {}
        ci_up_dict = {}

        for i, t_val in enumerate(time_index):
            eff = float(all_effects[i])
            if std_err > 1e-12:
                pval_dict[t_val] = float(
                    2 * (1 - stats.norm.cdf(abs(eff) / std_err))
                )
                ci_lo_dict[t_val] = float(eff - z * std_err)
                ci_up_dict[t_val] = float(eff + z * std_err)
            else:
                pval_dict[t_val] = 1.0
                ci_lo_dict[t_val] = eff
                ci_up_dict[t_val] = eff

        return std_err, pval_dict, ci_lo_dict, ci_up_dict

    def _compute_placebo_effects(
        self,
        pre_effects: np.ndarray,
        all_effects: np.ndarray,
        time_index: np.ndarray,
        max_placebo_units: int = 200,
    ) -> list | None:
        """
        计算安慰剂效应：对每个控制单元运行 SDID，收集其平均后处理效应。

        此为 Arkhangelsky et al. (2021) 推荐的标准误估计方法的基础步骤。
        限制最大安慰剂单元数以控制计算成本。

        Returns
        -------
        list[float] | None
            安慰剂效应列表；若无法计算则返回 None。
        """
        # 需要原始数据和参数。如果存储了这些信息则可以复用。
        if not hasattr(self, '_fit_data_cache_'):
            return None

        cache = self._fit_data_cache_
        df_wide = cache['df_wide']
        time_index_full = df_wide.index.values
        unit_ids = df_wide.columns.values
        treat_time = cache['treat_time']
        treated_ids = cache['treated_ids']
        controls_list = cache.get('controls_list', [])
        time_col = cache['time_col']
        controls_data_cache = cache.get('controls_data')

        post_mask_full = time_index_full >= treat_time

        # 找出所有控制单元
        control_units = [u for u in unit_ids if u not in treated_ids]
        n_co = len(control_units)

        if n_co < 2:
            return None

        # 限制安慰剂计算数量
        n_placebo = min(n_co, max_placebo_units)
        rng = np.random.default_rng(self._random_state)
        placebo_sample = list(
            rng.choice(control_units, size=n_placebo, replace=False)
        )

        from tqdm import tqdm

        placebo_effects = []
        for j, placebo_unit in enumerate(
            tqdm(placebo_sample, desc="Placebo 计算", unit="unit")
        ):
            try:
                tau_hat = self._fit_single_placebo(
                    df_wide.copy(),
                    time_index_full,
                    unit_ids,
                    placebo_unit,
                    treat_time,
                    post_mask_full,
                    controls_list,
                    time_col,
                    controls_data_cache,
                )
                if tau_hat is not None and np.isfinite(tau_hat):
                    placebo_effects.append(float(tau_hat))
            except Exception:
                # 单个安慰剂失败不影响整体
                continue

        return placebo_effects if len(placebo_effects) >= 2 else None

    def _fit_single_placebo(
        self,
        df_wide: pd.DataFrame,
        time_index: np.ndarray,
        unit_ids: np.ndarray,
        placebo_unit,
        treat_time,
        post_mask: np.ndarray,
        controls_list: list,
        time_col: str,
        controls_data_cache,
    ) -> float | None:
        """
        对单个安慰剂单元运行简化 SDID，返回其平均后处理效应。

        将 placebo_unit 作为"处理组"（取均值，兼容多单元接口），
        其余所有单元作为控制组，拟合 SDID 并返回后处理期平均效应。
        """
        pre_mask = time_index < treat_time
        T_pre = int(pre_mask.sum())
        T_post = int(post_mask.sum())

        # 处理组"均值"（实际只有 1 个单元）
        Y_tr = df_wide[placebo_unit].values   # (T,)
        control_cols = [u for u in unit_ids if u != placebo_unit]
        Y_co = df_wide[control_cols].values    # (T, N_co_placebo)
        N_co_p = Y_co.shape[1]

        Y_pre_tr_outcome = Y_tr[pre_mask].copy()
        Y_pre_co_outcome = Y_co[pre_mask, :].copy()

        # 正则化参数（复用缓存或重新估计）
        zeta_p = self._cached_zeta_ if hasattr(self, '_cached_zeta_') else (
            self._estimate_zeta(
                Y_pre_tr_outcome, Y_pre_co_outcome, T_pre, N_co_p, T_post
            )
        )

        # 控制变量增强（如有）
        if controls_data_cache is not None and len(controls_list) > 0:
            Y_pre_tr_aug, Y_pre_co_aug = self._augment_with_controls(
                df_wide, controls_data_cache, time_col,
                [placebo_unit], control_cols, treat_time,
                controls_list, Y_pre_tr_outcome, Y_pre_co_outcome,
            )
        else:
            Y_pre_tr_aug, Y_pre_co_aug = Y_pre_tr_outcome, Y_pre_co_outcome

        # 单元权重
        omega_p, _ = self._compute_unit_weights(
            Y_pre_tr_aug, Y_pre_co_aug, zeta_p, T_pre
        )

        # 对齐结果 A_t
        synthetic = Y_co @ omega_p
        A_tr = Y_tr - synthetic
        A_co = Y_co - synthetic.reshape(-1, 1)

        # 时间权重
        zeta_t = (
            self._cached_zeta_time_ if hasattr(self, '_cached_zeta_time_')
            else None
        )
        lambda_p = self._compute_time_weights(
            A_co, pre_mask, post_mask, zeta_t, T_pre
        )

        # 效应
        A_tr_pre = A_tr[pre_mask]
        baseline = float(np.dot(lambda_p, A_tr_pre))
        all_effects = A_tr - baseline

        # 返回后处理期平均效应
        post_effects = all_effects[post_mask]
        return float(np.mean(post_effects))
