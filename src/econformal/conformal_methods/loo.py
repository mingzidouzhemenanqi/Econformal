"""
LOO Conformal Inference — 留一法共形推断实现
=============================================

算法原理
--------
LOO (Leave-One-Out) 共形推断对处理前时间点进行留一训练，将所有 LOO 模型
在剩余处理前时点上的 |effect| 汇集成 nonconformity score 分布，从而在保持
数据利用效率的同时提供稳健的不确定性量化。

  1. LOO 训练: 逐个剔除处理前时间点，在剩余处理前数据上拟合计量模型，
     收集该模型在所有剩余处理前时点上的 |effect| 作为 nonconformity score。
  2. 校准: 对所有汇集的 LOO 残差排序，取调整分位数 q。
  3. 预测: 在全量数据上拟合最终模型，各处理后时点区间 = effect ± q。

与 Split Conformal 的核心区别在于：Split 做一次切分（训练/校准），
LOO 做 k 次留一训练（k = 处理前时间点数），每次留出一个时点
并从剩余时点提取残差。这避免了单次切分的随机性，数据利用更充分，
但计算代价更高（k 次拟合 vs 2 次拟合）。

数学定义
--------
令 T_pre = {t₀, t₁, ..., t_{k-1}} 为处理前所有时间点。

对每个 t_j ∈ T_pre：
  - 构造 LOO 训练集: data_{-j} = { (i, t) | (t < treat_time ∧ t ≠ t_j) ∨ t == treat_time }
  - 拟合计量模型 μ̂_{-j} = fit_econmodel(data_{-j})
  - 提取剩余处理前时点的 |effect| 作为 scores:
      scores_j = { |μ̂_{-j}(t).effect| : t < treat_time ∧ μ̂_{-j}(t).effect ≠ 0 }

汇集所有 LOO 的 scores:
  all_scores = scores_0 ∪ scores_1 ∪ ... ∪ scores_{k-1}

调整分位数（有限样本覆盖保证）:
  n = len(all_scores)
  k_idx = ⌈(n + 1) · coverage⌉
  q = sorted(all_scores)[k_idx - 1]

预测区间（处理后时期）:
  CI_t = [effect_t - q,  effect_t + q]   for t ≥ treat_time
  其中 effect_t 来自在全量数据上拟合的最终模型。

设计说明：为什么不用被剔除时点本身计算残差
------------------------------------------
标准 Jackknife+ 的残差定义为 R_j = |Y_j - μ̂_{-j}(X_j)|，即留一模型对被剔除
样本的预测误差。但计量模型 (DID/SC) 的 fit_econmodel() 只能在输入数据存在的
时间点上产生效应估计——它无法对训练数据之外的时间点做预测。因此本实现改为：
每个 LOO 模型对**剩余处理前时点**的 |effect| 作为 nonconformity score，度量
该模型在零假设（无处理效应）下的自然波动。所有 LOO 的 scores 池化后共同
校准分位数 q。

覆盖性质
--------
本实现是对留一法共形推断在面板数据计量模型场景下的改编，与标准 Jackknife+
有三点关键差异：

1. 残差定义：标准 Jackknife+ 使用 R_j = |Y_j - μ̂_{-j}(X_j)|（留一模型对
   被剔除点的预测误差），本实现改为汇集 LOO 模型在剩余处理前时点上的
   |effect|。这是因为计量模型的 fit_econmodel() 只能对训练数据中的时间点
   产生效应估计，无法对被剔除时点做外推预测。

2. 区间构造：标准 Jackknife+ 对每个测试点分别收集所有 LOO 模型的预测
   {μ̂_{-i}(x) ± R_i} 取分位数，区间可非对称且宽度可变。本实现使用全量
   模型估计的 effect ± 单一分位数 q，区间对称等宽。这是因为计量模型的
   "预测"是按时间参数化的，无法让 LOO 模型对任意时点独立预测。

3. 数据假设：标准 Jackknife+ 假设 i.i.d. 可交换性。本实现处理的是时间序列
   面板数据，处理前时点之间存在时序依赖，可交换性仅为近似。

基于以上差异，标准 Jackknife+ 的 P(Y ∈ CI) ≥ 1 - 2α 有限样本保证
不直接适用于本实现。实际覆盖性质取决于处理前时点数 k、计量模型类型、
以及时序依赖强度。在 k ≥ 5 且处理前时点无明显结构性断点的情况下，
仿真结果通常表明覆盖接近或超过名义水平（偏保守）。

与现有方法对比
--------------
  维度          | Split Conformal          | LOO Conformal               | Full Conformal
  --------------|--------------------------|-----------------------------|---------------------------
  训练方式      | 单次时序切分              | 逐个时点留一                | nulls 网格 + 置换检验
  拟合次数      | 2                        | k (处理前时点数)             | n_post × n_nulls
  数据利用      | 部分（训练+校准分离）     | 充分（每个时点都被留出一次） | 充分（但仅做增强变换）
  随机性        | 依赖单次切分位置          | 确定性（无随机切分）         | 确定性
  覆盖保证      | 近似 1-α                  | 仿真偏保守 (见覆盖性质)      | 精确有限样本
  区间形状      | 各时点等宽（同一个 q）    | 各时点等宽（同一个 q）       | 各时点宽度可不同
  适用场景      | 大数据、快速预览          | 中等数据、追求稳健           | 小样本、追求精确

数据流
------
  self.data (全量面板)
    │
    ├─ preprocess_data()
    │    ├─ pre_times = sorted(unique(time where time < treat_time))
    │    ├─ post_time_list = sorted(unique(time where time >= treat_time))
    │    └─ 返回 (pre_times, post_time_list)
    │
    ├─ fit(pre_times)
    │    │  对每个 t_j ∈ pre_times:
    │    ├─ loo_data = data[((time < treat_time) & (time != t_j)) | (time == treat_time)]
    │    ├─ results = econ_model.fit_econmodel(loo_data)
    │    ├─ scores_j = |results[(time < treat_time) & (|effect| > 1e-12)].effect|
    │    │
    │    ├─ self.loo_scores = sorted(all pooled scores)
    │    └─ self.quantile = loo_scores[ceil((n+1)*coverage) - 1]
    │
    ├─ predict(post_time_list)
    │    ├─ results = econ_model.fit_econmodel(self.data)
    │    ├─ post_effects = results[time >= treat_time].effect
    │    └─ [post_effects - q, post_effects + q]
    │
    └─ result_to_dataframe()
         └─ DataFrame(index=post_time_list, columns=[lower, upper])

注意事项
--------
1. 计算成本与处理前时间点数 k 成正比（k 次计量模型拟合）。当 k 较大时
   （如日频数据），建议使用 Split Conformal 以控制计算时间。
2. fit() 中的 LOO 训练是 embarrassingly parallel 的，未来可实现并行化。
3. 与 Split Conformal 一致，fit() 和 predict() 使用相同的计量模型接口
   fit_econmodel()，且最终区间各时点等宽（共享同一个 q）。
4. 若 pre_times 数量过少（<3），LOO 残差分布不稳定，此时 Split Conformal
   或 Full Conformal 可能更合适。
5. 若用户传入了 nulls 参数（Full Conformal 用），本方法会忽略并给出警告。
6. 每个 LOO 模型包含首个处理后时点 (time == treat_time) 以确保 Treat=1 行
   存在，但仅从处理前时点提取 nonconformity scores，避免处理后效应污染校准。
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from .conformal_base import ConformalBase


class Conformal(ConformalBase):
    """LOO (Leave-One-Out) 共形推断。

    对处理前每个时间点进行 Leave-One-Out 训练，汇集所有 LOO 模型在剩余
    处理前时点上的 |effect| 作为 nonconformity score 池，用调整分位数
    作为区间半宽，生成处理后时期的共形预测区间。

    方法定位
    --------
    本实现更准确的描述是 "LOO-Calibrated Conformal Inference"：用留一法
    替代 Split Conformal 的单次切分来做校准，从而消除切分位置的随机性。
    与 Split Conformal 共享相同的区间构造方式（effect ± 单一分位数 q），
    但校准过程更稳健。用户可通过 self.loo_scores 检查汇集的 nonconformity
    score 分布。

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
        占位参数，LOO 不使用 (保留与 Full Conformal 接口兼容)。
    controls_col : list, optional
        控制变量列名列表。
    **kwargs
        传递给 ConformalBase 的额外参数。

    Attributes
    ----------
    pre_times : np.ndarray
        处理前所有时间点（已排序）。
    post_time_list : np.ndarray
        处理后所有时间点（已排序）。
    loo_scores : np.ndarray
        所有 LOO 模型汇集的 nonconformity scores (|effect|，已排序)，供调试。
    quantile : float
        从 LOO 残差池计算的调整分位数。
    conformal_interval : pd.DataFrame
        最终共形预测区间，index=处理后时间，columns=[lower, upper]。
    """

    def __init__(self, econ_model, data: pd.DataFrame, time: str, id: str,
                 y_col: str, treat_col: str, coverage: float,
                 nulls: list = None,
                 controls_col: list = None,
                 econ_results: pd.DataFrame = None,
                 **kwargs):
        """初始化 LOO 实例。

        逻辑：
        - 若用户传入了 nulls，发出警告（本方法不使用 nulls）
        - 接收 econ_results（已由外层 _econ_fit() 计算的全量拟合结果），
          predict() 可直接复用，避免重复拟合
        - 委托 ConformalBase.__init__ 完成公共初始化：
            存储列名、生成置信区间列名、提取 target_id_list 和 treat_time
        """
        if controls_col is None:
            controls_col = []

        # LOO 不使用 nulls 参数，若用户传入则提示
        if nulls is not None:
            warnings.warn(
                f"LOO 不使用 nulls 参数，传入的 nulls (长度={len(nulls)}) 将被忽略。"
                f"如需使用 nulls 网格搜索，请选择 conformal_model='full'。"
            )

        # 保存已计算的全量拟合结果，predict() 可直接复用
        self.econ_results = econ_results

        super().__init__(
            econ_model=econ_model, data=data, time=time, id=id,
            y_col=y_col, treat_col=treat_col,
            coverage=coverage, controls_col=controls_col,
            **kwargs
        )

    # =========================================================================
    # 主入口
    # =========================================================================

    def compute_conformal_interval(self):
        """LOO 推断主流程。

        串联四步，与 Split/Full Conformal 的 compute_conformal_interval()
        结构一致：

        0. preprocess_data()   — 提取 pre_times 和 post_time_list
        1. fit(pre_times)      — LOO 训练 + 计算调整分位数 q
        2. predict(post_time_list) — 全量训练 + 生成预测区间
        3. result_to_dataframe()   — 格式化为标准 DataFrame

        Returns
        -------
        pd.DataFrame
            共形预测区间，index=处理后时间，
            columns=[f"{cov}%_conformal_lower", f"{cov}%_conformal_upper"]。
        """
        # 步骤 0: 数据预处理 — 提取 pre_times 和 post_time_list
        pre_times, post_time_list = self.preprocess_data()

        # 步骤 1: LOO 训练 + 计算调整分位数
        self.fit(pre_times)

        # 步骤 2: 全量训练 + 生成预测区间
        confidence_interval = self.predict(post_time_list)

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
                "处理前时期为空，无法进行 LOO 推断。"
                "请确保数据中包含处理前的时间点 (time < treat_time)。"
            )

        # 存储为实例属性，供调试
        self.pre_times = np.array(pre_times)

        # 处理前时点过少时发出警告：LOO 残差分布将不稳定，区间可能极宽
        if len(self.pre_times) < 2:
            raise ValueError(
                f"处理前时点仅有 {len(self.pre_times)} 个，LOO Conformal 至少需要 2 个"
                f"（至少 1 个训练点 + 1 个被留出的校准点）。"
                f"请使用 Full Conformal 或增加处理前时点数量。"
            )

        # 提取处理后时间列表 (predict 阶段需要)
        post_time_list = sorted(
            self.data.loc[self.data[self.time] >= self.treat_time, self.time].unique()
        )

        if len(post_time_list) == 0:
            raise ValueError(
                "处理后时期为空，无法进行 LOO 推断。"
                "请确保数据中包含处理后时期 (time >= treat_time)。"
            )

        return self.pre_times.copy(), np.array(post_time_list)

    # =========================================================================
    # 步骤 1: LOO 训练 + 计算调整分位数
    # =========================================================================

    def fit(self, pre_times):
        """对每个处理前时点做 Leave-One-Out，汇集成调整分位数 q。

        逻辑：
        对每个 t_j ∈ pre_times:
          1. loo_data = data[((time < treat_time) & (time != t_j)) | (time == treat_time)]
             → 剔除时点 t_j 的处理前行，但保留首个处理后时点 (确保 Treat=1 行存在)
          2. results = econ_model.fit_econmodel(loo_data, ...)
             → 在 LOO 数据上拟合计量模型
          3. scores_j = |results[(time < treat_time) & (effect != 0)].effect|
             → 提取该 LOO 模型在剩余处理前时点上的 |effect| 作为 nonconformity scores
             → 排除 effect==0 的时点（通常是参考期，其 effect 恒为 0，无信息量）

        收集所有 scores_j，排序后取调整分位数：
          all_scores = sorted(scores_0 ∪ scores_1 ∪ ... ∪ scores_{k-1})
          k_idx = ceil((n + 1) * self.coverage)
          self.quantile = all_scores[k_idx - 1]

        Parameters
        ----------
        pre_times : np.ndarray
            处理前所有唯一时间值（已排序），来自 preprocess_data()。

        Side Effects
        ------------
        self.loo_scores : np.ndarray
            LOO nonconformity scores（已排序），供调试和诊断。
        self.quantile : float
            调整分位数，供 predict() 构造区间。

        Raises
        ------
        ValueError
            若所有 LOO 残差均为 NaN 或无法计算（如计量模型省略了所有基准期）。
        """
        all_scores = []

        # 预计算不变的布尔 mask，避免 LOO 循环内重复扫描全量数据
        is_pre = self.data[self.time] < self.treat_time
        is_first_post = self.data[self.time] == self.treat_time

        for t_j in tqdm(pre_times, desc='LOO 训练'):

            # 构造 LOO 数据: 剔除时点 t_j 的处理前行 + 首个处理后时点
            loo_mask = (is_pre & (self.data[self.time] != t_j)) | is_first_post
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
            super()._validate_econ_results(results, context=f'LOO(t_j={t_j})')

            # 提取该 LOO 模型在剩余处理前时点上的 |effect|
            # 排除 effect≈0 的时点（参考期/无信息量），使用自适应容差
            tol = self._adaptive_tol(results['effect'].to_numpy())
            pre_mask = (
                (results[self.time] < self.treat_time) &
                (np.abs(results['effect']) > tol)
            )
            scores_from_loo = results.loc[pre_mask, 'effect'].abs().to_numpy()

            if len(scores_from_loo) > 0:
                all_scores.extend(scores_from_loo.tolist())

        if len(all_scores) == 0:
            raise ValueError(
                "所有 LOO 模型均未产生有效的 nonconformity score。"
                "可能原因：处理前时点过少，或计量模型将所有处理前时点的 effect 估计为 0。"
            )

        # 排序并存储
        self.loo_scores = np.sort(all_scores)

        # 计算调整分位数 (有限样本覆盖保证)
        # 公式: k_idx = ⌈(n+1)·coverage⌉, q = sorted_scores[k_idx - 1]
        n = len(self.loo_scores)
        k_idx = int(np.ceil((n + 1) * self.coverage))
        self.quantile = float(self.loo_scores[min(k_idx, n) - 1])

    # =========================================================================
    # 步骤 2: 生成预测区间
    # =========================================================================

    def predict(self, post_time_list):
        """生成处理后各时点的预测区间。

        优先复用外层 _econ_fit() 已计算好的 self.econ_results，
        避免重复拟合全量数据。若不可用则现场拟合。

        逻辑：
        1. 获取全量拟合结果（复用或现场拟合）
        2. 按 post_time_list 顺序提取各时点的效应估计
        3. CI_t = [effect_t - self.quantile, effect_t + self.quantile]
           → 所有时点共享同一个 q（等宽区间）

        Parameters
        ----------
        post_time_list : np.ndarray
            处理后时间值列表（已排序），来自 preprocess_data()。

        Returns
        -------
        np.ndarray
            形状 (n_post, 2)，每行为 [lower, upper]。

        Raises
        ------
        RuntimeError
            若尚未调用 fit() 计算分位数（self.quantile 不存在）。
        ValueError
            若结果中缺少某些处理后时点的效应估计。
        """
        if not hasattr(self, 'quantile'):
            raise RuntimeError(
                "尚未计算分位数，请先调用 fit() 再调用 predict()。"
            )

        # 优先复用已有的全量拟合结果，避免重复计算
        if self.econ_results is not None:
            results = self.econ_results
        else:
            results = self.econ_model.fit_econmodel(
                data=self.data,
                time=self.time,
                id=self.id,
                y_col=self.y_col,
                treat_col=self.treat_col,
                coverage=self.coverage,
                controls_col=self.controls_col,
                **self._econ_kwargs,
            )

        # 校验返回结果格式
        super()._validate_econ_results(results, context='predict')

        # 按 post_time_list 顺序提取效应估计
        # 使用 post_time_list 而非 results[time >= treat_time] 的原始顺序，
        # 确保与 result_to_dataframe 的 index 严格对齐
        results_indexed = results.set_index(self.time)
        missing = set(post_time_list) - set(results_indexed.index)
        if missing:
            raise ValueError(
                f"全量拟合结果中缺少以下处理后时点的效应估计: {sorted(missing)}。"
                f"treat_time={self.treat_time}。"
            )

        post_effects = results_indexed.loc[post_time_list, 'effect'].to_numpy()

        if len(post_effects) == 0:
            raise ValueError(
                "全量拟合未能产生处理后时期的效应估计。"
                f"treat_time={self.treat_time}，结果中的时间值: "
                f"{sorted(results[self.time].unique())}"
            )

        # 构造预测区间（对称等宽）
        lower = post_effects - self.quantile
        upper = post_effects + self.quantile

        return np.column_stack((lower, upper))

    # =========================================================================
    # 步骤 3: 格式化为 DataFrame
    # =========================================================================

    def result_to_dataframe(self, confidence_interval, time_list):
        """将预测区间数组转换为标准 DataFrame。

        逻辑：
        - 以 time_list 为 index
        - 列名为 [self.ci_lower_col, self.ci_upper_col]
          （如 "90%_conformal_lower", "90%_conformal_upper"）
        - 与 Split/Full Conformal 输出格式完全一致，确保 _merge_results() 兼容

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

