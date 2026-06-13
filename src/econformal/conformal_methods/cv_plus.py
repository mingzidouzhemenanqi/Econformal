"""
CV+ (Cross-Validation Plus) Conformal Inference — K 折交叉验证非对称共形推断实现
================================================================================

算法原理
--------
CV+ (Barber et al., 2021, "Predictive inference with the jackknife+") 是 Jackknife+
的 K 折交叉验证推广。与 Jackknife+ 对每个处理前时点做留一法（LOO）不同，CV+ 将
处理前时点划分为 K 个折（fold），对每个折训练一个模型（剔除该折数据），从而将
模型拟合次数从 n_pre 降低到 K（默认 5），大幅降低计算成本。

  1. 折划分: 将处理前时间点 T_pre = {t₀, t₁, ..., t_{m-1}} 划分为 K 个折
     F₀, F₁, ..., F_{K-1}。支持两种划分策略：
     - 'random': 随机打乱后均匀分配（标准 CV+，与 i.i.d. 可交换性假设一致）
     - 'block': 按时间顺序连续分块（保留时序结构，更适合时间序列面板数据）

  2. 折训练: 对每个折 k = 0..K-1：
     a. 构造训练数据 data_{-k} = 全部数据 − 折 k 的处理前行（保留全部处理后行）
     b. 在 data_{-k} 上拟合计量模型 μ̂_{-k}
     c. 计算折残差 R_k（nonconformity score）：
        - SC 模式（折中时点出现在模型结果中）:
          R_k = mean(|effect_{-k}(t)| for t in F_k)  —— 折内时点残差的均值
        - DID 模式（折中时点不在结果中）:
          R_k = mean(|effect| on remaining pre-periods)  —— 剩余处理前时点均值
     d. 对每个处理后时点 t，提取效应估计 effect_{-k}(t)

  3. 区间构造（每个处理后时点 t 独立计算）:
     - lower_candidates_k(t) = effect_{-k}(t) - R_k
     - upper_candidates_k(t) = effect_{-k}(t) + R_k
     - LB(t) = α-分位数 of {lower_candidates_k(t)}   (α = 1 - coverage)
     - UB(t) = (1-α)-分位数 of {upper_candidates_k(t)}

与 Jackknife+ 的对比
---------------------
  维度          | Jackknife+                  | CV+
  --------------|-----------------------------|---------------------------
  训练方式      | 逐个时点 LOO (留一法)        | K 折交叉验证
  拟合次数      | m = len(pre_times)           | K (默认 5)
  残差粒度      | 按时点: R_j = |effect(t_j)|   | 按折: R_k = mean(折内残差)
  效应收集      | 每模型存各时点 effect_{-j}(t) | 每折模型存各时点 effect_{-k}(t)
  区间形状      | 各时点独立，可非对称         | 各时点独立，可非对称
  计算效率      | O(m) 次拟合                  | O(K) 次拟合 (K << m)
  覆盖保证      | P(Y ∈ CI) ≥ 1 − 2α           | P(Y ∈ CI) ≥ 1 − 2α (近似)

覆盖性质
--------
标准 CV+ 在 i.i.d. 可交换性假设下具有 P(Y ∈ CI) ≥ 1 − 2α 的有限样本保证。
本实现将该方法适配到面板数据计量模型场景：

1. 残差定义适配：SC 模型可从折外数据预测被剔除折内时点的效应，因此可使用
   R_k = mean(|effect_{-k}(t)| for t in F_k)，与标准 CV+ 的折残差定义一致。
   DID 模型无法对训练数据外的时间点做预测，因此使用剩余处理前时点的平均
   |effect| 作为 R_k 的代理。

2. 数据假设：面板数据中处理前时点之间存在时序依赖，可交换性仅为近似。
   使用 'block' 策略可更好地保持时序结构，但折数较少时折间异质性可能增大。

3. 在 K ≥ 5 且处理前时点数充足（≥ 10）的情况下，仿真结果通常表明覆盖接近
   或超过名义水平。

数据流
------
  self.data (全量面板)
    │
    ├─ preprocess_data()
    │    ├─ pre_times = sorted(unique(time where time < treat_time))
    │    ├─ post_time_list = sorted(unique(time where time >= treat_time))
    │    ├─ k = min(cv_folds, len(pre_times))
    │    ├─ folds = build_folds(pre_times, k, strategy)  # random 或 block
    │    └─ 返回 (folds, post_time_list)
    │
    ├─ fit(folds, post_time_list)
    │    │  对每个 fold ∈ folds:
    │    ├─ train_data = data[~fold_pre_mask]  # 剔除该折的处理前行
    │    ├─ results = econ_model.fit_econmodel(train_data)
    │    ├─ R_k = _compute_fold_residual(results, fold_times)
    │    ├─ self.cvplus_residuals[k] = R_k
    │    │  对每个 t ∈ post_time_list:
    │    └─ self.cvplus_effects[t][k] = effect_{-k}(t)
    │    └─ 存储 R_k 分布统计: rk_mean, rk_std, rk_min, rk_max
    │
    ├─ predict()
    │    │  对每个 t ∈ self.post_time_list:
    │    ├─ lower_cands = {effect_{-k}(t) - R_k}  (过滤 NaN)
    │    ├─ upper_cands = {effect_{-k}(t) + R_k}
    │    ├─ LB(t) = α-quantile(lower_cands)
    │    └─ UB(t) = (1-α)-quantile(upper_cands)
    │
    └─ result_to_dataframe()
         └─ DataFrame(index=post_time_list, columns=[lower, upper])

注意事项
--------
1. 计算成本与折数 K 成正比（K 次计量模型拟合）。K 默认值为 5，在效率和
   稳定性之间取得平衡。处理前时点较多时（如日频数据），CV+ 比 JK+ 显著更快。
2. 折划分可通过 random_state 参数控制随机种子（默认 42），确保结果可复现。
3. 若处理前时点数少于 cv_folds，实际折数会自动缩减为处理前时点数（此时
   CV+ 退化为 JK+），同时发出警告。
4. 若用户传入了 nulls 参数（Full Conformal 用），本方法会忽略并给出警告。
5. 每个折模型包含全部处理后时点 (time >= treat_time) 以确保能对所有
   处理后时点产生效应估计。
6. predict() 不使用全量模型的 econ_results 构造区间，区间完全基于折模型预测。
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .conformal_base import ConformalBase


class Conformal(ConformalBase):
    """CV+ (Cross-Validation Plus) 共形推断。

    将处理前时间点划分为 K 个折进行交叉验证训练，保留每个折模型的残差 R_k
    和各处理后时点的效应估计 effect_{-k}(t)，对每个处理后时点独立构造可变的、
    可能非对称的共形预测区间。

    CV+ 是 Jackknife+ 的 K 折推广，核心优势是计算效率：
    Jackknife+ 需要拟合 m = n_pre 个模型，CV+ 只需拟合 K 个（默认 5）。
    当处理前时点较多时（如日频/周频面板数据），计算成本大幅降低。

    与 Jackknife+ 的本质区别
    -------------------------
    JK+ 对每个处理前时点留一，残差粒度是单个时点（R_j = |effect(t_j)|）。
    CV+ 对每个折留一，残差粒度是折内时点的聚合（R_k = mean(|effect(t)| for t in fold)）。
    当 K = len(pre_times) 时，每个折恰好一个时点，CV+ 等价于 JK+。

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
    cv_folds : int, optional
        交叉验证折数，默认 5。必须 >= 2。若超过处理前时点数则自动缩减。
    cv_strategy : str, optional
        折划分策略，默认 'random'。
        - 'random': 随机打乱处理前时点后均匀分配（标准 CV+）。
        - 'block': 按时间顺序连续分块（保留时序结构）。
    random_state : int, optional
        随机种子，默认 42。仅在 cv_strategy='random' 时生效，
        用于控制 shuffle 的结果可复现性。
    nulls : list, optional
        占位参数，CV+ 不使用 (保留与 Full Conformal 接口兼容)。
    controls_col : list, optional
        控制变量列名列表。
    econ_results : pd.DataFrame, optional
        外层 _econ_fit() 已计算的全量拟合结果，predict() 中不使用
        （区间构造完全基于折模型预测），保留以维持接口兼容性。
    **kwargs
        传递给 ConformalBase 的额外参数（如 split_rate，将被静默吸收）。

    Attributes
    ----------
    cv_folds : int
        用户请求的折数。
    k : int
        实际使用的折数（可能因数据不足而缩减）。
    cv_strategy : str
        折划分策略 ('random' 或 'block')。
    folds : list of np.ndarray
        每个折包含的处理前时间值列表，长度 = k。
    post_time_list : np.ndarray
        处理后所有时间点（已排序）。
    cvplus_residuals : list of float
        每个折模型的 nonconformity score R_k，长度 = k。
    cvplus_effects : dict
        key = 处理后时点 t, value = list of float (长度 = k)，
        其中 cvplus_effects[t][k] = effect_{-k}(t)。
    conformal_interval : pd.DataFrame
        最终共形预测区间，index=处理后时间，columns=[lower, upper]。
    """

    def __init__(self, econ_model, data: pd.DataFrame, time: str, id: str,
                 y_col: str, treat_col: str, coverage: float,
                 cv_folds: int = 5,
                 cv_strategy: str = 'random',
                 random_state: int = 42,
                 nulls: list = None,
                 controls_col: list = None,
                 econ_results: pd.DataFrame = None,
                 **kwargs):
        """初始化 CV+ 实例。

        逻辑：
        - 校验 cv_folds 和 cv_strategy 参数
        - 若用户传入了 nulls，发出警告（本方法不使用 nulls）
        - 接收 econ_results 以维持接口兼容性（但 predict() 不使用）
        - 委托 ConformalBase.__init__ 完成公共初始化：
            存储列名、生成置信区间列名、提取 target_id_list 和 treat_time
        """
        if controls_col is None:
            controls_col = []

        # ---- 校验 cv_folds ----
        if not isinstance(cv_folds, int) or cv_folds < 2:
            raise ValueError(
                f"cv_folds 必须为 >= 2 的整数，当前值: {cv_folds}。"
                f"CV+ 至少需要 2 折才能进行交叉验证。"
            )
        self.cv_folds = cv_folds

        # ---- 校验 cv_strategy ----
        _VALID_STRATEGIES = {'random', 'block'}
        if cv_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"cv_strategy 必须为 {_VALID_STRATEGIES} 之一，当前值: '{cv_strategy}'。"
            )
        self.cv_strategy = cv_strategy

        # ---- 存储随机种子 ----
        self.random_state = random_state

        # 初始化 fit() 产出为 None，供 predict() 做明确的 None 检查
        self.cvplus_residuals = None
        self.cvplus_effects = None
        self.post_time_list = None

        # 回退警告去重标志：确保 _compute_fold_residual 的 fallback 警告最多触发一次
        self._fallback_warned = False

        # CV+ 不使用 nulls 参数，若用户传入则提示
        if nulls is not None:
            warnings.warn(
                f"CV+ 不使用 nulls 参数，传入的 nulls (长度={len(nulls)}) 将被忽略。"
                f"如需使用 nulls 网格搜索，请选择 conformal_model='full'。"
            )

        # 保存全量拟合结果（接口兼容，predict() 不使用）
        self.econ_results = econ_results

        # 委托基类完成公共初始化
        super().__init__(
            econ_model=econ_model, data=data, time=time, id=id,
            y_col=y_col, treat_col=treat_col,
            coverage=coverage, controls_col=controls_col,
            **kwargs
        )

        # ---- 兼容性警告：CV+ + DID ----
        econ_module = type(self.econ_model).__module__
        if econ_module.split('.')[-1] == 'did':
            warnings.warn(
                "CV+ (Cross-Validation Plus) 与 DID 组合：R_k 使用剩余处理前时点的 "
                "mean(|effect|) 作为代理折残差，而非标准 CV+ 要求的 "
                "held-out 折残差 (被剔除时点本身的预测误差均值)。"
                "这是因为 DID 的 fit_econmodel() 无法对训练数据之外的时间点 "
                "做预测。\n"
                "  → 本实现是标准 CV+ 的适配版，P(Y in CI) >= 1-2alpha "
                "的有限样本保证不直接适用。\n"
                "  → 如需精确覆盖保证，请使用 conformal_model='full'。\n"
                "  → SC 模型支持完整的 CV+ 残差定义，可优先考虑 "
                "conformal_model='cv+' + econ_model='sc'。"
            )

    # =========================================================================
    # 主入口
    # =========================================================================

    def compute_conformal_interval(self):
        """CV+ 推断主流程。

        串联四步，与 Split/Full/LOO/JK+ 的 compute_conformal_interval()
        结构一致：

        0. preprocess_data()                  — 提取 pre_times、构建折、获取 post_time_list
        1. fit(folds, post_time_list)         — K 折训练 + 存储 R_k 和 effect_{-k}(t)
        2. predict()                          — 构造 CV+ 区间
        3. result_to_dataframe()              — 格式化为标准 DataFrame

        Returns
        -------
        pd.DataFrame
            共形预测区间，index=处理后时间，
            columns=[f"{cov}%_conformal_lower", f"{cov}%_conformal_upper"]。
        """
        # 步骤 0: 数据预处理 — 提取 pre_times、构建折、获取 post_time_list
        folds, post_time_list = self.preprocess_data()
        self.post_time_list = post_time_list  # 统一来源，供 fit/predict 共用

        # 步骤 1: K 折训练 + 存储每折残差和效应
        self.fit(folds, post_time_list)

        # 步骤 2: 构造 CV+ 区间
        confidence_interval = self.predict()

        # 步骤 3: 格式化为 DataFrame
        self.conformal_interval = self.result_to_dataframe(
            confidence_interval, post_time_list)

        return self.conformal_interval

    # =========================================================================
    # 步骤 0: 数据预处理 — 提取时间点并构建折
    # =========================================================================

    def preprocess_data(self):
        """提取处理前/后时间点列表，并按策略构建 K 个折。

        逻辑：
        1. 提取 pre_times = 所有 time < treat_time 的唯一值，升序
        2. 提取 post_time_list = 所有 time >= treat_time 的唯一值，升序
        3. 根据 cv_strategy 将 pre_times 划分为 k = min(cv_folds, len(pre_times)) 个折：
           - 'random': 用固定种子 shuffle 后均匀切分
           - 'block': 保持原始排序，按连续块均匀切分
        4. 边界校验

        Returns
        -------
        folds : list of np.ndarray
            每个元素是一个折包含的处理前时间值数组，长度为 k。
        post_time_list : np.ndarray
            处理后所有唯一时间值（已排序）。

        Raises
        ------
        ValueError
            若处理前或处理后时期为空。
        """
        # ---- 提取时间列，一次扫描完成处理前/后时间点分类 ----
        time_col = self.data[self.time]
        pre_times = sorted(time_col[time_col < self.treat_time].unique())
        post_time_list = sorted(time_col[time_col >= self.treat_time].unique())

        # ---- 验证处理前时期 ----
        if len(pre_times) == 0:
            raise ValueError(
                "处理前时期为空，无法进行 CV+ 推断。"
                "请确保数据中包含处理前的时间点 (time < treat_time)。"
            )

        self.pre_times = np.array(pre_times)

        # 处理前时点过少时发出警告
        if len(self.pre_times) < 2:
            raise ValueError(
                f"处理前时点仅有 {len(self.pre_times)} 个，CV+ 至少需要 2 个"
                f"（至少 1 个训练点 + 1 个校准点）。"
                f"请使用 Full Conformal 或增加处理前时点数量。"
            )
        if len(self.pre_times) < 4:
            warnings.warn(
                f"处理前时点仅有 {len(self.pre_times)} 个（< 4），"
                f"CV+ 的分位数估计将不稳定。"
                f"建议使用 Split Conformal 或 Full Conformal 替代。"
            )

        # ---- 验证处理后时期 ----
        if len(post_time_list) == 0:
            raise ValueError(
                "处理后时期为空，无法进行 CV+ 推断。"
                "请确保数据中包含处理后时期 (time >= treat_time)。"
            )

        # ---- 确定实际折数 k ----
        m = len(self.pre_times)
        k = min(self.cv_folds, m)
        self.k = k

        if k < self.cv_folds:
            warnings.warn(
                f"处理前时点数 ({m}) 少于请求的折数 ({self.cv_folds})，"
                f"已将折数缩减为 {k}。"
                f"此时 CV+ 退化为 JK+ 模式（每折一个时点）。"
            )

        # ---- 按策略构建折 ----
        if self.cv_strategy == 'random':
            # 随机打乱（使用用户指定的 random_state 确保可复现），然后均匀切分
            rng = np.random.RandomState(self.random_state)
            shuffled = self.pre_times.copy()
            rng.shuffle(shuffled)
            folds = self._split_into_folds(shuffled, k)
        else:  # 'block'
            # 保持原始时间顺序，按连续块均匀切分
            folds = self._split_into_folds(self.pre_times.copy(), k)

        self.folds = folds

        # ---- 折质量检查 ----
        fold_sizes = [len(f) for f in folds]
        if any(s < 2 for s in fold_sizes):
            warnings.warn(
                f"存在大小仅为 {min(fold_sizes)} 的折（< 2），"
                f"折内仅一个时点时 CV+ 的折残差估计可能不稳定。"
            )
        if max(fold_sizes) > 3 * min(fold_sizes) and k > 2:
            warnings.warn(
                f"折大小严重不均: min={min(fold_sizes)}, max={max(fold_sizes)}。"
                f"这可能影响 CV+ 的覆盖性质。"
            )

        return folds, np.array(post_time_list)

    @staticmethod
    def _split_into_folds(arr, k):
        """将数组均匀切分为 k 个折。

        使用"先分配商、再分配余数"的策略确保各折大小之差不超过 1：
        - 每个折的基础大小为 n // k
        - 前 n % k 个折多分配一个元素

        Parameters
        ----------
        arr : np.ndarray
            待切分的数组（可能已被 shuffle）。
        k : int
            折数。

        Returns
        -------
        list of np.ndarray
            长度为 k 的列表，每个元素为一个折。
        """
        n = len(arr)
        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[:n % k] += 1  # 前 n % k 个折多一个元素

        folds = []
        start = 0
        for size in fold_sizes:
            folds.append(arr[start:start + size])
            start += size

        return folds

    # =========================================================================
    # 步骤 1: K 折训练 + 存储残差与效应
    # =========================================================================

    def fit(self, folds, post_time_list):
        """对每个折做留一折训练，存储每折的 R_k 和 effect_{-k}(t)。

        逻辑：
        对每个折 k = 0..K-1:
          1. train_data = data[~fold_k_pre_mask]
             → 剔除折 k 中的所有处理前行，但保留全部处理后时点
             （确保 Treat=1 行存在且能对所有处理后时点产生效应估计）
          2. results = econ_model.fit_econmodel(train_data, ...)
             → 在折外数据上拟合计量模型
          3. 计算 R_k 通过 _compute_fold_residual:
             - SC 模式：折内时点出现在 results 中 →
               R_k = mean(|effect_{-k}(t)| for t in fold_k)
             - DID 模式：折内时点不在 results 中 →
               R_k = mean(|effect| on remaining pre-periods)
          4. 对每个处理后时点 t ∈ post_time_list:
             提取 effect_{-k}(t) 并存入 self.cvplus_effects[t][k]
             （若该折模型未产生 t 时点的效应，存入 NaN）

        Parameters
        ----------
        folds : list of np.ndarray
            折列表，每个折包含该折的处理前时间值。
        post_time_list : np.ndarray
            处理后所有唯一时间值（已排序），来自 preprocess_data()。

        Side Effects
        ------------
        self.cvplus_residuals : list of float
            每个折模型的 nonconformity score R_k，长度 = k。
        self.cvplus_effects : dict
            key = 处理后时点 t, value = [effect_{-0}(t), ..., effect_{-(k-1)}(t)]。
        self.rk_mean, self.rk_std, self.rk_min, self.rk_max : float
            R_k 分布摘要统计，供诊断使用。

        Warnings
        --------
        UserWarning
            若所有 R_k ≈ 0（CV+ 退化为效应分位数区间）。
        """
        # ---- 初始化存储结构 ----
        self.cvplus_residuals = []
        self.cvplus_effects = {t: [] for t in post_time_list}

        # 重置回退警告标志（支持同一实例多次调用 compute_conformal_interval）
        self._fallback_warned = False

        # ---- 预提取时间列，避免循环内 K 次重复列访问 ----
        time_series = self.data[self.time]

        # ---- K 折训练循环 ----
        for fold_idx, fold_times in enumerate(
            tqdm(folds, desc=f'CV+ 训练 (K={self.k})')
        ):
            # 构造折训练数据：剔除折 k 的所有处理前行 + 全部处理后行
            # fold_times 仅含处理前时点，isin 自动限定在处理前行，无需额外 mask
            fold_pre_mask = time_series.isin(fold_times)
            fold_train_mask = ~fold_pre_mask
            fold_data = self.data[fold_train_mask]

            # 在折外数据上拟合计量模型（透传用户 kwargs 确保内外拟合规格一致）
            results = self.econ_model.fit_econmodel(
                data=fold_data,
                time=self.time,
                id=self.id,
                y_col=self.y_col,
                treat_col=self.treat_col,
                coverage=self.coverage,
                controls_col=self.controls_col,
                **self._econ_kwargs,
            )

            # 校验返回结果格式
            super()._validate_econ_results(
                results, context=f'CV+ fold {fold_idx}')

            # ---- 计算折残差 R_k ----
            R_k = self._compute_fold_residual(results, fold_times)
            self.cvplus_residuals.append(R_k)

            # ---- 提取各处理后时点的效应估计 ----
            # 每折构建一次 {time: effect} 字典，避免 n_post 次 .loc 扫描
            time_to_effect = dict(zip(results[self.time], results['effect']))
            for t in post_time_list:
                if t in time_to_effect:
                    self.cvplus_effects[t].append(
                        float(time_to_effect[t]))
                else:
                    # 该折模型未对该时点产生效应估计
                    # （如 DID 的相对时间映射偏移导致某些时点被省略）
                    self.cvplus_effects[t].append(np.nan)

        # ---- 后循环验证与统计 ----
        rk_arr = np.array(self.cvplus_residuals)

        # 存储 R_k 分布统计供诊断
        self.rk_mean = float(np.mean(rk_arr))
        self.rk_std = float(np.std(rk_arr))
        self.rk_min = float(np.min(rk_arr))
        self.rk_max = float(np.max(rk_arr))

        # 所有 R_k ≈ 0 时发出警告：CV+ 退化为普通分位数
        if self.rk_max < 1e-12:
            warnings.warn(
                f"所有 {len(rk_arr)} 个折模型的 R_k 均为 0.0，"
                f"CV+ 已退化为效应分位数区间。"
                f"可能原因：处理前时点过少、计量模型将全部处理前 effect 估计为 0、"
                f"或面板数据缺乏处理前变异。"
                f"建议检查计量模型在处理前时点上的拟合质量。"
            )

    # =========================================================================
    # 折残差计算辅助方法
    # =========================================================================

    def _compute_fold_residual(self, results, fold_times):
        """计算单个折模型的 nonconformity score R_k。

        自动检测计量模型类型以选择合适的残差计算方式：

        **SC 模式**（折内时点出现在 results 中）:
          对被剔除折中的每个时点 t ∈ fold_times，若 t 出现在 results 中且
          effect ≠ 0，则收集 |effect_{-k}(t)| 作为直接残差。
          R_k = mean(收集到的直接残差)
          这与标准 CV+ 的折残差定义一致。

        **DID 模式**（折内时点不在 results 中）:
          DID 的 fit_econmodel() 无法对训练数据之外的时间点做预测，
          被剔除的折内时点不会出现在结果中。此时回退到使用剩余处理前时点的
          平均 |effect| 作为 R_k 的代理。
          R_k = mean(|effect| on remaining pre-periods with |effect| > 1e-12)

        **混合模式**（折内部分时点在 results 中但 effect=0，如 SC 参考期）:
          仅收集 effect ≠ 0 的直接残差。若无有效直接残差，回退到 DID 模式。

        Parameters
        ----------
        results : pd.DataFrame
            fit_econmodel() 的返回结果，必须包含 time 列和 effect 列。
        fold_times : np.ndarray
            当前被剔除折包含的处理前时点值。

        Returns
        -------
        float
            折级别的 nonconformity score R_k。
        """
        # ---- 尝试 SC 模式：向量化收集折内时点的直接残差 ----
        time_values = results[self.time].values

        # 将折内时点统一转换为与 results 时间列相同的 dtype
        fold_typed = np.array(fold_times).astype(time_values.dtype)

        # 一次 isin 定位折内所有出现在结果中的时点行
        fold_mask = np.isin(time_values, fold_typed)

        # 自适应容差：随 effect 尺度缩放
        tol = self._adaptive_tol(results['effect'].to_numpy())

        if fold_mask.any():
            # 批量提取 |effect|，过滤掉 effect≈0 的参考期时点
            direct_effects = results.loc[fold_mask, 'effect'].abs()
            direct_residuals = direct_effects[direct_effects > tol].tolist()

            if direct_residuals:
                return float(np.mean(direct_residuals))

        # ---- 回退到 DID 模式：使用剩余处理前时点的平均 |effect| ----
        pre_mask = (
            (results[self.time] < self.treat_time) &
            (np.abs(results['effect']) > tol)
        )
        pre_effects = results.loc[pre_mask, 'effect'].abs().to_numpy()

        if len(pre_effects) > 0:
            return float(np.mean(pre_effects))

        # ---- 完全无法计算残差 ----
        if not self._fallback_warned:
            warnings.warn(
                f"CV+ 无法为部分折提取有效的 nonconformity score，"
                f"R_k 设为 0.0（首个触发折: {list(fold_times)}）。"
                f"可能原因：所有处理前时点的 effect 均为 0（参考期），"
                f"或计量模型未产生任何处理前效应估计。"
                f"这可能影响 CV+ 区间的覆盖性质。"
                f"（此警告仅显示一次，后续相同情况将静默处理）"
            )
            self._fallback_warned = True
        return 0.0

    # =========================================================================
    # 步骤 2: 构造 CV+ 预测区间
    # =========================================================================

    def predict(self):
        """对每个处理后时点独立构造 CV+ 预测区间。

        不使用全量模型的 econ_results，区间完全基于折模型的预测。
        使用 fit() 中存储的 self.cvplus_effects 和 self.cvplus_residuals，
        以及 compute_conformal_interval() 设置的 self.post_time_list。

        对每个 t ∈ self.post_time_list:
          1. 收集有效的 (effect_{-k}(t), R_k) 对，过滤 NaN
             （NaN 出现在某些折模型未对该时点产生效应估计时）
          2. lower_candidates_k = effect_{-k}(t) - R_k
          3. upper_candidates_k = effect_{-k}(t) + R_k
          4. 排序 lower/upper candidates
          5. LB(t) = α-分位数 of lower_candidates（α = 1 - coverage）
          6. UB(t) = (1-α)-分位数 of upper_candidates

        分位数使用有限样本调整公式: k_idx = ⌈(n+1)·p⌉，与 JK+/LOO/Split 一致。
        这确保即使在折数较少时也能提供合理的覆盖（偏保守）。

        Returns
        -------
        np.ndarray
            形状 (n_post, 2)，每行为 [lower, upper]。

        Raises
        ------
        RuntimeError
            若尚未调用 fit()（self.cvplus_residuals 或 self.cvplus_effects 不存在）。
        ValueError
            若某处理后时点在所有折模型中均无有效效应估计。
        """
        # ---- 前置条件检查 ----
        if self.cvplus_residuals is None:
            raise RuntimeError(
                "尚未进行 K 折训练，请先调用 fit() 再调用 predict()。"
            )
        if self.cvplus_effects is None:
            raise RuntimeError(
                "尚未收集折模型效应估计，请先调用 fit() 再调用 predict()。"
            )
        if self.post_time_list is None:
            raise RuntimeError(
                "尚未设置 post_time_list，请通过 compute_conformal_interval() 调用。"
            )

        alpha = 1.0 - self.coverage       # 下尾分位数水平
        residuals_arr = np.array(self.cvplus_residuals)  # shape (k,)

        all_lower = []
        all_upper = []

        # ---- 按处理后时点逐个构造区间 ----
        for t in self.post_time_list:
            # 收集该时点所有折模型的效应估计
            effects = np.array(self.cvplus_effects[t])

            # 过滤 NaN（某些折模型未对该时点产生估计）
            valid_mask = ~np.isnan(effects)
            valid_effects = effects[valid_mask]
            valid_residuals = residuals_arr[valid_mask]

            n = len(valid_effects)
            if n == 0:
                raise ValueError(
                    f"处理后时点 t={t} 在所有 {len(self.cvplus_residuals)} 个"
                    f"折模型中均无有效效应估计。"
                    f"请检查计量模型是否正确拟合了该时点。"
                )

            # 有效预测数过少时警告
            if n < 5:
                warnings.warn(
                    f"处理后时点 t={t} 仅有 {n} 个有效折模型预测（< 5），"
                    f"CV+ 分位数估计不稳定。"
                )

            # ---- 构造候选区间边界 ----
            lower_candidates = valid_effects - valid_residuals
            upper_candidates = valid_effects + valid_residuals

            # ---- 排序并取有限样本调整分位数 ----
            sorted_lower = np.sort(lower_candidates)
            sorted_upper = np.sort(upper_candidates)

            # LB: α-分位数（下尾），对称保护上下溢
            # 公式: k_idx = ⌈(n+1)·α⌉, LB = sorted[k_idx - 1]
            k_lower = int(np.ceil((n + 1) * alpha))
            lb = sorted_lower[min(max(k_lower - 1, 0), n - 1)]

            # UB: (1-α)-分位数（上尾）= coverage-分位数
            # 公式: k_idx = ⌈(n+1)·coverage⌉, UB = sorted[k_idx - 1]
            k_upper = int(np.ceil((n + 1) * self.coverage))
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
        - 与 Split/Full/LOO/JK+ 输出格式完全一致，确保 _merge_results() 兼容

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
