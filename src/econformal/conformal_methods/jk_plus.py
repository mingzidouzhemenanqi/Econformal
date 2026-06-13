"""
Jackknife+ Conformal Inference — 留一法非对称共形推断实现
========================================================

算法原理
--------
Jackknife+ (Barber et al., 2021) 是对标准 Jackknife (留一法) 共形推断的改进。
与 LOO Conformal 的核心区别在于：LOO 将所有留一模型的残差池化为一个全局分位数，
构造等宽对称区间；Jackknife+ 保留每个留一模型的残差 R_j 和该模型对各测试时点的
预测 effect_{-j}(t)，对每个测试时点独立构造可变的、可能非对称的区间。

  1. LOO 训练: 逐个剔除处理前时间点 t_j，在剩余数据上拟合计量模型 μ̂_{-j}。
  2. 残差提取: 对每个 μ̂_{-j}，计算该模型的 nonconformity score R_j：
     - SC 模式（t_j 在结果中）: R_j = |effect_{-j}(t_j)|，即留一模型对剔除时点的预测误差
     - DID 模式（t_j 不在结果中）: R_j = mean(|effect| on remaining pre-periods)
  3. 效应收集: 对每个处理后时点 t，收集所有 LOO 模型在该时点的效应估计
     effect_{-j}(t) (j = 0..k-1)。
  4. 区间构造（每个处理后时点 t 独立计算）:
     - lower_candidates_j(t) = effect_{-j}(t) - R_j
     - upper_candidates_j(t) = effect_{-j}(t) + R_j
     - LB(t) = α-分位数 of {lower_candidates_j(t)}   (α = 1 - coverage)
     - UB(t) = 1-α 分位数 of {upper_candidates_j(t)}

与 LOO Conformal 的对比
------------------------
  维度          | LOO Conformal               | Jackknife+
  --------------|-----------------------------|---------------------------
  残差存储      | 池化为全局列表 all_scores    | 每模型单独 R_j
  效应存储      | 仅用全量模型的 effect(t)     | 每模型存储 effect_{-j}(t)
  区间中心      | 全量模型 effect(t)            | LOO 模型 effect_{-j}(t) 直接参与
  区间宽度      | 等宽 (同一个 q)               | 各时点独立变化
  区间对称性    | 对称 (±q)                    | 可非对称
  覆盖保证      | 仿真偏保守                    | P(Y ∈ CI) ≥ 1 - 2α (标准 Jackknife+)

覆盖性质
--------
标准 Jackknife+ 在 i.i.d. 可交换性假设下具有 P(Y ∈ CI) ≥ 1 - 2α 的有限样本
保证。本实现将该方法适配到面板数据计量模型场景：

1. 残差定义适配：SC 模型可从留一数据预测所有时点（包括被剔除时点），因此可直接
   使用 R_j = |effect_{-j}(t_j)|，与标准 Jackknife+ 一致。DID 模型无法对训练
   数据外的时间点做预测，因此使用剩余处理前时点的平均 |effect| 作为 R_j。

2. 数据假设：面板数据中处理前时点之间存在时序依赖，可交换性仅为近似。
   实际覆盖性质取决于处理前时点数 k、计量模型类型和时序依赖强度。

3. 在 k ≥ 5 且无结构性断点的情况下，仿真结果通常表明覆盖接近或超过名义水平。

数据流
------
  self.data (全量面板)
    │
    ├─ preprocess_data()
    │    ├─ pre_times = sorted(unique(time where time < treat_time))
    │    ├─ post_time_list = sorted(unique(time where time >= treat_time))
    │    └─ 返回 (pre_times, post_time_list)
    │
    ├─ fit(pre_times, post_time_list)
    │    │  对每个 t_j ∈ pre_times:
    │    ├─ loo_data = data[((time < treat_time) & (time != t_j)) | (time >= treat_time)]
    │    ├─ results = econ_model.fit_econmodel(loo_data)
    │    ├─ R_j = auto_detect_residual(results, t_j)
    │    ├─ self.jkplus_residuals[j] = R_j
    │    │  对每个 t ∈ post_time_list:
    │    └─ self.jkplus_effects[t][j] = effect_{-j}(t)
    │    └─ 存储 R_j 分布统计: rj_mean, rj_std, rj_min, rj_max
    │
    ├─ predict()
    │    │  对每个 t ∈ self.post_time_list:
    │    ├─ lower_cands = {effect_{-j}(t) - R_j}  (过滤 NaN)
    │    ├─ upper_cands = {effect_{-j}(t) + R_j}
    │    ├─ LB(t) = α-quantile(lower_cands)
    │    └─ UB(t) = (1-α)-quantile(upper_cands)
    │
    └─ result_to_dataframe()
         └─ DataFrame(index=post_time_list, columns=[lower, upper])

注意事项
--------
1. 计算成本与处理前时间点数 k 成正比（k 次计量模型拟合），与 LOO 相同。
2. fit() 中的 LOO 训练是 embarrassingly parallel 的，未来可实现并行化。
3. 若 pre_times 数量过少（<4），Jackknife+ 的分位数估计不稳定。
4. 若用户传入了 nulls 参数（Full Conformal 用），本方法会忽略并给出警告。
5. 每个 LOO 模型包含全部处理后时点 (time >= treat_time) 以确保能对所有
   处理后时点产生效应估计，同时仅剔除一个处理前时点 t_j。
6. predict() 不使用全量模型的 econ_results 构造区间，区间完全基于 LOO 模型预测。
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .conformal_base import ConformalBase


class Conformal(ConformalBase):
    """Jackknife+ 共形推断。

    对处理前每个时间点进行 Leave-One-Out 训练，保留每个 LOO 模型的残差 R_j
    和各处理后时点的效应估计 effect_{-j}(t)，对每个处理后时点独立构造可变的、
    可能非对称的共形预测区间。

    与 LOO Conformal 的本质区别
    -------------------------
    LOO 将残差池化后取全局分位数，构造等宽对称区间（effect ± q）。
    Jackknife+ 保留每个 LOO 模型的独立信息，对每个测试时点 t 分别
    取 {effect_{-j}(t) ± R_j} 的分位数，区间宽度可随时点变化且可非对称。
    这使得 Jackknife+ 能更好地适应异方差性和模型在不同时点上的不确定性差异。

    Parameters
    ----------
    econ_model : object
        已初始化的计量模型实例 (Econometric)，提供 fit_econmodel() 方法。
    data : pd.DataFrame
        完整面板数据，含处理前和处理后时期。
    time : str
        时间列名。
    id : str
        个体标识列名。
    y_col : str
        因变量列名。
    treat_col : str
        处理指示列名 (0/1 二值)。
    coverage : float
        目标覆盖率，取值范围 (0, 1)。例如 0.9 表示 90% 置信区间。
    nulls : list, optional
        占位参数，Jackknife+ 不使用 (保留与 Full Conformal 接口兼容)。
    controls_col : list, optional
        控制变量列名列表。
    econ_results : pd.DataFrame, optional
        外层 _econ_fit() 已计算的全量拟合结果，predict() 中不使用
        （区间构造完全基于 LOO 模型预测），保留以维持接口兼容性。
    **kwargs
        传递给 ConformalBase 的额外参数（如 split_rate，将被静默吸收）。

    Attributes
    ----------
    pre_times : np.ndarray
        处理前所有时间点（已排序）。
    post_time_list : np.ndarray
        处理后所有时间点（已排序）。
    jkplus_residuals : list of float
        每个 LOO 模型的 nonconformity score R_j，长度 = k。
    jkplus_effects : dict
        key = 处理后时点 t, value = list of float (长度 = k)，
        其中 jkplus_effects[t][j] = effect_{-j}(t)。
    conformal_interval : pd.DataFrame
        最终共形预测区间，index=处理后时间，columns=[lower, upper]。
    """

    def __init__(self, econ_model, data: pd.DataFrame, time: str, id: str,
                 y_col: str, treat_col: str, coverage: float,
                 nulls: list = None,
                 controls_col: list = None,
                 econ_results: pd.DataFrame = None,
                 **kwargs):
        """初始化 Jackknife+ 实例。

        逻辑：
        - 若用户传入了 nulls，发出警告（本方法不使用 nulls）
        - 接收 econ_results 以维持接口兼容性（但 predict() 不使用）
        - 委托 ConformalBase.__init__ 完成公共初始化：
            存储列名、生成置信区间列名、提取 target_id_list 和 treat_time
        """
        if controls_col is None:
            controls_col = []

        # Jackknife+ 不使用 nulls 参数，若用户传入则提示
        if nulls is not None:
            warnings.warn(
                f"Jackknife+ 不使用 nulls 参数，传入的 nulls (长度={len(nulls)}) 将被忽略。"
                f"如需使用 nulls 网格搜索，请选择 conformal_model='full'。"
            )

        # 保存全量拟合结果（接口兼容，predict() 不使用）
        self.econ_results = econ_results

        super().__init__(
            econ_model=econ_model, data=data, time=time, id=id,
            y_col=y_col, treat_col=treat_col,
            coverage=coverage, controls_col=controls_col,
            **kwargs
        )

        # ---- 兼容性警告：JK+ + DID ----
        econ_module = type(self.econ_model).__module__
        if econ_module.split('.')[-1] == 'did':
            warnings.warn(
                "JK+ (Jackknife+) 与 DID 组合：R_j 使用剩余处理前时点的 "
                "mean(|effect|) 作为代理残差，而非标准 Jackknife+ 要求的 "
                "held-out 残差 (被剔除时点本身的预测误差)。"
                "这是因为 DID 的 fit_econmodel() 无法对训练数据之外的时间点 "
                "做预测。\n"
                "  → 本实现是标准 Jackknife+ 的适配版，P(Y in CI) >= 1-2alpha "
                "的有限样本保证不直接适用。\n"
                "  → 如需精确覆盖保证，请使用 conformal_model='full'。\n"
                "  → SC 模型支持完整的 Jackknife+ 残差定义，可优先考虑 "
                "conformal_model='jk+' + econ_model='sc'。"
            )

    # =========================================================================
    # 主入口
    # =========================================================================

    def compute_conformal_interval(self):
        """Jackknife+ 推断主流程。

        串联四步，与 Split/Full/LOO Conformal 的 compute_conformal_interval()
        结构一致：

        0. preprocess_data()                — 提取 pre_times 和 post_time_list
        1. fit(pre_times, post_time_list)   — LOO 训练 + 存储 R_j 和 effect_{-j}(t)
        2. predict()                        — 构造 Jackknife+ 区间
        3. result_to_dataframe()            — 格式化为标准 DataFrame

        Returns
        -------
        pd.DataFrame
            共形预测区间，index=处理后时间，
            columns=[f"{cov}%_conformal_lower", f"{cov}%_conformal_upper"]。
        """
        # 步骤 0: 数据预处理 — 提取 pre_times 和 post_time_list
        pre_times, post_time_list = self.preprocess_data()
        self.post_time_list = post_time_list  # 统一来源，供 fit/predict 共用

        # 步骤 1: LOO 训练 + 存储每模型残差和效应
        self.fit(pre_times, post_time_list)

        # 步骤 2: 构造 Jackknife+ 区间
        confidence_interval = self.predict()

        # 步骤 3: 格式化为 DataFrame
        self.conformal_interval = self.result_to_dataframe(
            confidence_interval, post_time_list)

        return self.conformal_interval

    # =========================================================================
    # 步骤 0: 数据预处理
    # =========================================================================

    def preprocess_data(self):
        """提取处理前和处理后时间点列表。

        逻辑：
        - pre_times = 所有 time < treat_time 的唯一值，升序
        - post_time_list = 所有 time >= treat_time 的唯一值，升序
        - 边界检查：两者均不可为空

        Returns
        -------
        pre_times : np.ndarray
            处理前所有唯一时间值（已排序）。
        post_time_list : np.ndarray
            处理后所有唯一时间值（已排序）。

        Raises
        ------
        ValueError
            若处理前或处理后时期为空。
        """
        # 提取处理前所有唯一时间，升序排列
        pre_times = sorted(
            self.data.loc[self.data[self.time] < self.treat_time, self.time].unique()
        )

        if len(pre_times) == 0:
            raise ValueError(
                "处理前时期为空，无法进行 Jackknife+ 推断。"
                "请确保数据中包含处理前的时间点 (time < treat_time)。"
            )

        # 存储为实例属性，供调试
        self.pre_times = np.array(pre_times)

        # 处理前时点过少时发出警告：分位数估计将不稳定
        if len(self.pre_times) < 2:
            raise ValueError(
                f"处理前时点仅有 {len(self.pre_times)} 个，Jackknife+ 至少需要 2 个"
                f"（至少 1 个训练点 + 1 个被留出的校准点）。"
                f"请使用 Full Conformal 或增加处理前时点数量。"
            )
        if len(self.pre_times) < 4:
            warnings.warn(
                f"处理前时点仅有 {len(self.pre_times)} 个（< 4），"
                f"Jackknife+ 的分位数估计将不稳定。"
                f"建议使用 Split Conformal 或 Full Conformal 替代。"
            )

        # 提取处理后时间列表 (predict 阶段需要)
        post_time_list = sorted(
            self.data.loc[self.data[self.time] >= self.treat_time, self.time].unique()
        )

        if len(post_time_list) == 0:
            raise ValueError(
                "处理后时期为空，无法进行 Jackknife+ 推断。"
                "请确保数据中包含处理后时期 (time >= treat_time)。"
            )

        return self.pre_times.copy(), np.array(post_time_list)

    # =========================================================================
    # 步骤 1: LOO 训练 + 存储残差与效应
    # =========================================================================

    def fit(self, pre_times, post_time_list):
        """对每个处理前时点做 Leave-One-Out，存储每模型的 R_j 和 effect_{-j}(t)。

        逻辑：
        对每个 t_j ∈ pre_times:
          1. loo_data = data[((time < treat_time) & (time != t_j)) | (time >= treat_time)]
             → 剔除时点 t_j 的处理前行，但保留全部处理后时点 (确保 Treat=1 行存在
                且能对所有处理后时点产生效应估计)
          2. results = econ_model.fit_econmodel(loo_data, ...)
             → 在 LOO 数据上拟合计量模型
          3. 计算 R_j 通过自动检测（SC vs DID）:
             - 若 t_j ∈ results[self.time]（SC 模式）:
               R_j = |effect_{-j}(t_j)|（直接 LOO 残差）
             - 否则（DID 模式）:
               R_j = mean(|effect| on remaining pre-periods)
          4. 对每个处理后时点 t ∈ post_time_list:
             提取 effect_{-j}(t) 并存入 self.jkplus_effects[t][j]
             （若该模型未产生 t 时点的效应，存入 NaN）

        Parameters
        ----------
        pre_times : np.ndarray
            处理前所有唯一时间值（已排序），来自 preprocess_data()。
        post_time_list : np.ndarray
            处理后所有唯一时间值（已排序），来自 preprocess_data()。

        Side Effects
        ------------
        self.jkplus_residuals : list of float
            每个 LOO 模型的 nonconformity score R_j，长度 = k。
        self.jkplus_effects : dict
            key = 处理后时点 t, value = [effect_{-0}(t), ..., effect_{-(k-1)}(t)]。
        self.rj_mean, self.rj_std, self.rj_min, self.rj_max : float
            R_j 分布摘要统计，供诊断使用。

        Warnings
        --------
        UserWarning
            若所有 R_j ≈ 0（Jackknife+ 退化为效应分位数区间）。
        """
        # 初始化存储结构
        self.jkplus_residuals = []
        self.jkplus_effects = {t: [] for t in post_time_list}

        # 预计算不变的布尔 mask，避免 LOO 循环内重复扫描全量数据
        is_pre = self.data[self.time] < self.treat_time
        is_post = self.data[self.time] >= self.treat_time

        for t_j in tqdm(pre_times, desc='Jackknife+ 训练'):

            # 构造 LOO 数据: 剔除时点 t_j 的处理前行 + 全部处理后时点
            # 注意：必须包含全部处理后时点（而非仅首个），因为 Jackknife+
            # 需要每个 LOO 模型对所有处理后时点产生效应估计。
            loo_mask = (is_pre & (self.data[self.time] != t_j)) | is_post
            loo_data = self.data[loo_mask]

            # 在 LOO 数据上拟合计量模型（透传用户 kwargs 确保内外拟合规格一致）
            results = self.econ_model.fit_econmodel(
                data=loo_data,
                time=self.time,
                id=self.id,
                y_col=self.y_col,
                treat_col=self.treat_col,
                coverage=self.coverage,
                controls_col=self.controls_col,
                **self._econ_kwargs,
            )

            # 校验返回结果格式
            super()._validate_econ_results(results, context=f'JK+(t_j={t_j})')

            # ---- 计算 R_j: 自动检测 SC vs DID 模式 ----
            R_j = self._compute_residual(results, t_j)

            self.jkplus_residuals.append(R_j)

            # ---- 提取各处理后时点的效应估计 ----
            for t in post_time_list:
                effect_rows = results.loc[results[self.time] == t, 'effect']
                if len(effect_rows) > 0:
                    self.jkplus_effects[t].append(float(effect_rows.values[0]))
                else:
                    # LOO 模型未对该时点产生效应估计（如 DID 的相对时间映射偏移）
                    self.jkplus_effects[t].append(np.nan)

        # ---- 后循环验证 ----
        # 注：preprocess_data() 已保证 pre_times 非空，循环至少执行一次，
        # 每次迭代必然 append R_j，因此 jkplus_residuals 不可能为空
        rj_arr = np.array(self.jkplus_residuals)

        # 存储 R_j 分布统计供诊断
        self.rj_mean = float(np.mean(rj_arr))
        self.rj_std = float(np.std(rj_arr))
        self.rj_min = float(np.min(rj_arr))
        self.rj_max = float(np.max(rj_arr))

        # 所有 R_j ≈ 0 时发出警告：Jackknife+ 退化为普通分位数
        if self.rj_max < 1e-12:
            warnings.warn(
                f"所有 {len(rj_arr)} 个 LOO 模型的 R_j 均为 0.0，"
                f"Jackknife+ 已退化为效应分位数区间。"
                f"可能原因：处理前时点过少、计量模型将全部处理前 effect 估计为 0、"
                f"或面板数据缺乏处理前变异。"
                f"建议检查计量模型在处理前时点上的拟合质量。"
            )

    def _compute_residual(self, results, t_j):
        """计算单个 LOO 模型的 nonconformity score R_j。

        自动检测计量模型类型：
        - 若 t_j 出现在 results 中（SC 模式）：R_j = |effect_{-j}(t_j)|
        - 若 t_j 不在 results 中（DID 模式）：R_j = mean(|effect| on remaining pre-periods)

        Parameters
        ----------
        results : pd.DataFrame
            fit_econmodel() 的返回结果。
        t_j : int or float
            当前被剔除的处理前时点值。

        Returns
        -------
        float
            Nonconformity score R_j。
        """
        # 提取剩余处理前时点的 |effect|（排除 effect≈0 的参考期），使用自适应容差
        tol = self._adaptive_tol(results['effect'].to_numpy())
        pre_mask = (
            (results[self.time] < self.treat_time) &
            (np.abs(results['effect']) > tol)
        )
        pre_effects = results.loc[pre_mask, 'effect'].abs().to_numpy()

        # 类型安全：将 t_j 转换为 results[self.time] 的 dtype 再比较，
        # 避免 numpy int64 vs Python int / float 的成员测试失败
        time_values = results[self.time].values
        t_j_typed = np.array([t_j]).astype(time_values.dtype)[0]
        t_j_present = np.any(time_values == t_j_typed)

        if t_j_present:
            # SC 模式：可直接对被剔除时点计算残差
            # SC 的 results 每行是一个个体在该时点的观测，同一时点所有行的 effect 相同
            effect_at_tj = results.loc[
                results[self.time] == t_j, 'effect'
            ].iloc[0]
            if abs(effect_at_tj) > tol:
                return float(abs(effect_at_tj))
            # effect≈0（如恰好是参考期）→ 回退到通用 fallback

        # 通用 fallback：无法对被剔除时点直接计算残差时（DID 模式 或 SC 参考期），
        # 使用剩余处理前时点的平均 |effect| 作为 R_j
        if len(pre_effects) > 0:
            return float(np.mean(pre_effects))
        else:
            mode = "SC (effect≈0)" if t_j_present else "DID"
            warnings.warn(
                f"JK+(t_j={t_j}): {mode} 模式下无法提取有效的 nonconformity score，"
                f"R_j 设为 0.0。这可能影响 Jackknife+ 区间的覆盖性质。"
            )
            return 0.0

    # =========================================================================
    # 步骤 2: 构造 Jackknife+ 预测区间
    # =========================================================================

    def predict(self):
        """对每个处理后时点独立构造 Jackknife+ 预测区间。

        不使用全量模型的 econ_results，区间完全基于 LOO 模型的预测。
        使用 fit() 中存储的 self.jkplus_effects 和 self.jkplus_residuals，
        以及 compute_conformal_interval() 设置的 self.post_time_list。

        对每个 t ∈ self.post_time_list:
          1. 收集有效的 (effect_{-j}(t), R_j) 对，过滤 NaN
          2. lower_candidates_j = effect_{-j}(t) - R_j
          3. upper_candidates_j = effect_{-j}(t) + R_j
          4. LB(t) = α-分位数 of lower_candidates（α = 1 - coverage）
          5. UB(t) = 1-α 分位数 of upper_candidates

        分位数使用有限样本调整公式: k_idx = ⌈(n+1)·p⌉，与 LOO/Split 一致。

        Returns
        -------
        np.ndarray
            形状 (n_post, 2)，每行为 [lower, upper]。

        Raises
        ------
        RuntimeError
            若尚未调用 fit()（self.jkplus_residuals 或 self.jkplus_effects 不存在）。
        ValueError
            若某处理后时点在所有 LOO 模型中均无有效效应估计。
        """
        if not hasattr(self, 'jkplus_residuals'):
            raise RuntimeError(
                "尚未进行 LOO 训练，请先调用 fit() 再调用 predict()。"
            )
        if not hasattr(self, 'jkplus_effects'):
            raise RuntimeError(
                "尚未收集 LOO 效应估计，请先调用 fit() 再调用 predict()。"
            )
        if not hasattr(self, 'post_time_list'):
            raise RuntimeError(
                "尚未设置 post_time_list，请通过 compute_conformal_interval() 调用。"
            )

        alpha = 1.0 - self.coverage
        residuals_arr = np.array(self.jkplus_residuals)

        all_lower = []
        all_upper = []

        for t in self.post_time_list:
            # 收集该时点所有 LOO 模型的效应估计
            effects = np.array(self.jkplus_effects[t])

            # 过滤 NaN（某些 LOO 模型未对该时点产生估计）
            valid_mask = ~np.isnan(effects)
            valid_effects = effects[valid_mask]
            valid_residuals = residuals_arr[valid_mask]

            n = len(valid_effects)
            if n == 0:
                raise ValueError(
                    f"处理后时点 t={t} 在所有 {len(self.jkplus_residuals)} 个"
                    f" LOO 模型中均无有效效应估计。"
                    f"请检查计量模型是否正确拟合了该时点。"
                )

            if n < 5:
                warnings.warn(
                    f"处理后时点 t={t} 仅有 {n} 个有效 LOO 预测（< 5），"
                    f"Jackknife+ 分位数估计不稳定。"
                )

            # 构造候选区间边界
            lower_candidates = valid_effects - valid_residuals
            upper_candidates = valid_effects + valid_residuals

            # 排序
            sorted_lower = np.sort(lower_candidates)
            sorted_upper = np.sort(upper_candidates)

            # 有限样本调整分位数
            # LB: α-分位数（下尾），UB: (1-α)-分位数（上尾）
            k_lower = int(np.ceil((n + 1) * alpha))
            k_upper = int(np.ceil((n + 1) * self.coverage))

            lb = sorted_lower[min(max(k_lower - 1, 0), n - 1)]
            ub = sorted_upper[min(k_upper - 1, n - 1)]

            all_lower.append(lb)
            all_upper.append(ub)

        return np.column_stack((all_lower, all_upper))

    # =========================================================================
    # 步骤 3: 格式化为 DataFrame
    # =========================================================================

    def result_to_dataframe(self, confidence_interval, time_list):
        """将预测区间数组转换为标准 DataFrame。

        逻辑：
        - 以 time_list 为 index
        - 列名为 [self.ci_lower_col, self.ci_upper_col]
          （如 "90%_conformal_lower", "90%_conformal_upper"）
        - 与 Split/Full/LOO Conformal 输出格式完全一致，确保 _merge_results() 兼容

        Parameters
        ----------
        confidence_interval : np.ndarray
            形状 (n, 2) 的数组，每行为 [lower, upper]。
        time_list : np.ndarray
            时间值列表，作为 DataFrame 的 index。

        Returns
        -------
        pd.DataFrame
            index = time_list，
            columns = [f"{cov}%_conformal_lower", f"{cov}%_conformal_upper"]。
        """
        return pd.DataFrame(
            confidence_interval,
            index=time_list,
            columns=[self.ci_lower_col, self.ci_upper_col]
        )
