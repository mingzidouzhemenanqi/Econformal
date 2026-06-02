"""
Split Conformal Inference — 拆分共形推断实现
================================================

算法原理
--------
Split Conformal 将处理前数据按时序切分为训练集和校准集：
  1. 训练集 (前 split_rate 占比的处理前时间) → 参与预测模型拟合
  2. 校准集 (后 (1-split_rate) 占比的处理前时间) → 计算 nonconformity scores
  3. 用校准集 scores 的调整分位数 q 构建处理后时期的预测区间

数学定义
--------
令 T_pre  = {t₀, t₁, ..., t_{k-1}} 为处理前所有时间点。
按 split_rate 比例切分 (索引方式):
  - 训练期:  time <  split_time   (前 split_rate 占比)
  - 校准期:  split_time ≤ time < treat_time   (后 1-split_rate 占比)

Nonconformity score (校准期):
  s_t = |effect_t|   for t ∈ 校准期
  (零假设下真实处理效应为 0，|effect| 量化了模型在该时点的自然波动)

调整分位数 (有限样本覆盖保证):
  n = len(cal_scores)
  k = ⌈(n + 1) · coverage⌉
  q = sorted(scores)[k - 1]

预测区间 (处理后时期):
  CI_t = [effect_t - q,  effect_t + q]   for t ≥ treat_time

模型强度对齐
------------
fit() 和 predict() 共享同一份 self.train_data（训练期 + 首个处理后时点）
作为拟合基础，分别拼接校准期数据和全量处理后数据。这确保两次计量模型
拟合的数据基础对等，校准期 nonconformity scores 与预测期效应的噪声水平
可比，避免因"fit 用全量处理前数据训练、predict 仅用训练期数据"导致的
分位数偏小、区间欠覆盖问题。

数据流
------
  self.data (全量面板)
    │
    ├─ preprocess_data()
    │    ├─ pre_times = unique(time where time < treat_time)
    │    ├─ split_idx = int(len(pre_times) * split_rate) → split_time = pre_times[split_idx]
    │    ├─ self.train_data = data[(time < split_time) | (time == treat_time)]
    │    ├─ cal_data = data[(split_time ≤ time) & (time < treat_time)]
    │    └─ post_time_list = unique(time where time >= treat_time)
    │
    ├─ fit(cal_data)
    │    ├─ fit_data = concat(self.train_data, cal_data)
    │    ├─ econ_model.fit_econmodel(fit_data) → results
    │    ├─ cal_scores = |results[split_time ≤ time < treat_time].effect|
    │    └─ quantile = sorted(cal_scores)[ceil((n+1)*coverage) - 1]
    │
    ├─ predict(post_time_list)
    │    ├─ post_data = data[time > treat_time]
    │    ├─ fit_data = concat(self.train_data, post_data)
    │    ├─ econ_model.fit_econmodel(fit_data) → results
    │    ├─ post_effects = results[time >= treat_time].effect
    │    └─ [post_effects - q, post_effects + q]
    │
    └─ result_to_dataframe()
         └─ DataFrame(index=post_time_list, columns=[lower, upper])

与 Full Conformal 对比
----------------------
  维度          | Full Conformal              | Split Conformal
  --------------|-----------------------------|---------------------------
  区间构造      | nulls 网格 + 置换检验 p 值   | 校准期 |effect| 分位数
  计算代价      | O(n_post × n_nulls) 次拟合  | O(2) 次拟合
  区间形状      | 各时点宽度可不同            | 各时点等宽 (同一个 q)
  覆盖保证      | 精确有限样本                | 近似有限样本 (依赖 exchangeability)
  适用场景      | 小样本、追求精确            | 大数据、快速预览

注意事项
--------
1. 计量模型 (DID/SC) 的 fit_econmodel() 是 fit + predict 一体化接口，
   无法分离训练和预测步骤。fit() 和 predict() 共享 self.train_data
   作为共同基础，确保两次拟合的模型强度对等。
2. self.train_data 包含首个处理后时点，以确保 Treat=1 行存在，
   计量模型能正确识别处理时间和处理组。
3. predict() 排除校准期数据，确保预测模型不会"看到"校准集信息。
"""

import numpy as np
import pandas as pd
import warnings
from .conformal_base import ConformalBase


class Conformal(ConformalBase):
    """Split Conformal 共形推断。

    将处理前时期按时序切分为训练集和校准集：
    - 训练集 (self.train_data) 作为 fit() 和 predict() 的共同拟合基础
    - 校准集计算 nonconformity scores（|effect|）
    - 校准集 scores 的调整分位数 q 作为区间半宽
    - 处理后区间 = effect ± q

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
    split_rate : float, default=0.7
        训练集在处理前时期中的占比，取值范围 (0, 1)。
        0.7 表示前 70% 处理前时间用于训练，后 30% 用于校准。
    nulls : list, optional
        占位参数，Split Conformal 不使用 (保留与 Full Conformal 接口兼容)。
    controls_col : list, optional
        控制变量列名列表。
    **kwargs
        传递给 ConformalBase 的额外参数。

    Attributes
    ----------
    split_time : int
        训练集与校准集的切分时间点 (取自数据中的实际时间值)。
    train_data : pd.DataFrame
        fit() 和 predict() 共享的训练基础数据。
    quantile : float
        从校准集 nonconformity scores 计算的调整分位数。
    cal_scores : np.ndarray
        校准期的 nonconformity scores (|effect|，已排序)，供调试。
    conformal_interval : pd.DataFrame
        最终共形预测区间，index=处理后时间，columns=[lower, upper]。
    """

    def __init__(self, econ_model, data: pd.DataFrame, time: str, id: str,
                 y_col: str, treat_col: str, coverage: float,
                 split_rate: float = 0.7, nulls: list = None,
                 controls_col: list = None, **kwargs):
        """初始化 Split Conformal 实例。

        保存 split_rate，其余列名/数据/coverage 等委托给 ConformalBase.__init__
        统一处理：存储列名、生成置信区间列名、提取 target_id_list 和 treat_time。
        """
        if controls_col is None:
            controls_col = []

        # 校验 split_rate 范围，防止极端值导致模型退化
        if not 0.1 <= split_rate <= 0.9:
            raise ValueError(
                f"split_rate 应在 [0.1, 0.9] 范围内，当前值为 {split_rate}。"
                f"过小 (<0.1) 会导致训练集不足，过大 (>0.9) 会导致校准集不可靠。"
            )

        # Split Conformal 不使用 nulls 参数，若用户传入则提示
        if nulls is not None:
            warnings.warn(
                f"Split Conformal 不使用 nulls 参数，传入的 nulls (长度={len(nulls)}) 将被忽略。"
                f"如需使用 nulls 网格搜索，请选择 conformal_model='full'。"
            )

        self.split_rate = split_rate

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
        """Split Conformal 推断主流程。

        串联数据预处理 → 计算分位数 → 生成区间 → 格式化输出，
        与 Full Conformal 的 compute_conformal_interval() 结构一致。

        Returns
        -------
        pd.DataFrame
            共形预测区间，index=处理后时间，
            columns=[f"{cov}%_conformal_lower", f"{cov}%_conformal_upper"]。
        """
        # 步骤 0: 数据预处理 — 提取 train_data + cal_data + post_time_list
        cal_data, post_time_list = self.preprocess_data()

        # 步骤 1: 拟合 + 计算调整分位数 — 从校准期 |effect| 得到 q
        self.fit(cal_data)

        # 步骤 2: 生成预测区间 — 处理后 effect ± q
        confidence_interval = self.predict(post_time_list)

        # 步骤 3: 格式化为 DataFrame — 与 Full Conformal 输出格式一致
        self.conformal_interval = self.result_to_dataframe(
            confidence_interval, post_time_list)

        return self.conformal_interval

    # =========================================================================
    # 步骤 0: 数据预处理
    # =========================================================================

    def preprocess_data(self):
        """按时序切分处理前数据，构造训练集和校准集。

        切分逻辑:
          pre_times = {t₀, t₁, ..., t_{k-1}}  全部 < treat_time
          split_idx = int(len(pre_times) * split_rate)
          split_time = pre_times[split_idx]
          - 训练期: time < split_time
          - 校准期: split_time ≤ time < treat_time

        self.train_data 的构造:
          训练期数据 + 首个处理后时间点 (time == treat_time)
          → fit() 和 predict() 共享的拟合基础
          → 确保两次计量模型拟合的数据基础对等，模型强度一致

        cal_data 的构造:
          校准期数据 (split_time ≤ time < treat_time)
          → 仅用于 fit() 提取 nonconformity scores
          → 不参与 predict() 的模型拟合

        Returns
        -------
        cal_data : pd.DataFrame
            校准期数据，仅 fit() 使用。
        post_time_list : list
            处理后时间值列表 (已排序)。

        Side Effects
        ------------
        self.split_time : int
            训练/校准切分时间点。
        self.train_data : pd.DataFrame
            fit() 和 predict() 共享的训练基础数据。
        """
        # 提取处理前所有唯一时间，升序排列
        pre_times = sorted(
            self.data.loc[self.data[self.time] < self.treat_time, self.time].unique()
        )

        # 边界检查: 必须存在处理前时期
        if len(pre_times) == 0:
            raise ValueError(
                "处理前时期为空，无法进行 Split Conformal 推断。"
                "请确保数据中包含处理前的时间点 (time < treat_time)。"
            )

        # 按索引确定切分点，比例精确匹配 split_rate
        # 例: pre_times=[2000,2001,2002,2003,2004], split_rate=0.7
        #     split_idx = int(5 * 0.7) = 3 → split_time=2003
        #     训练期: <2003 → [2000,2001,2002] (3/5=60%≈70%)
        #     校准期: ≥2003 → [2003,2004] (2/5=40%≈30%)
        split_idx = int(len(pre_times) * self.split_rate)
        # 确保校准集至少保留 1 个时点
        split_idx = min(split_idx, len(pre_times) - 1)
        # 确保训练集至少保留 1 个时点
        split_idx = max(split_idx, 1)
        self.split_time = pre_times[split_idx]

        # 构造训练基础数据:
        #   训练期 (time < split_time) + 首个处理后时点 (time == treat_time)
        #   作用: (1) 确保 Treat=1 行存在，计量模型可正确识别处理时间
        #         (2) 作为 fit() 和 predict() 的共同拟合基础
        self.train_data = self.data[
            (self.data[self.time] < self.split_time) |
            (self.data[self.time] == self.treat_time)
        ]

        # 边界检查: 训练集不能为空
        if len(self.train_data) == 0:
            raise ValueError(
                f"训练集为空。split_time={self.split_time}，"
                f"没有满足 time < split_time 的行。"
                f"当前 split_rate={self.split_rate}，请增大该值。"
            )

        # 提取校准期数据 (仅用于 fit 提取 nonconformity scores)
        cal_data = self.data[
            (self.data[self.time] >= self.split_time) &
            (self.data[self.time] < self.treat_time)
        ]

        # 边界检查: 校准集不能为空
        if len(cal_data) == 0:
            raise ValueError(
                f"校准集为空。split_time={self.split_time}，"
                f"没有满足 split_time ≤ time < treat_time={self.treat_time} 的行。"
                f"当前 split_rate={self.split_rate}，请减小该值。"
            )

        # 提取处理后时间列表 (predict 阶段需要)
        post_time_list = sorted(
            self.data.loc[self.data[self.time] >= self.treat_time, self.time].unique()
        )

        return cal_data, post_time_list

    # =========================================================================
    # 步骤 1: 计算调整分位数 (共用 self.train_data 基础)
    # =========================================================================

    def fit(self, cal_data):
        """计算校准期的调整分位数。

        用 train_data + cal_data 拟合计量模型，提取校准期 |effect|
        作为 nonconformity scores，计算调整分位数 q。

        Parameters
        ----------
        cal_data : pd.DataFrame
            校准期数据 (来自 preprocess_data)。

        Side Effects
        ------------
        self.cal_scores : np.ndarray
            校准期 nonconformity scores（已排序），供调试。
        self.quantile : float
            调整分位数。
        """
        # 拟合模型并获取完整结果
        results = self._fit_model(cal_data, caller='fit')

        # 提取校准期 |effect| 作为 nonconformity scores
        cal_mask = (
            (results[self.time] >= self.split_time) &
            (results[self.time] < self.treat_time)
        )
        scores = results.loc[cal_mask, 'effect'].abs().to_numpy()

        if len(scores) == 0:
            raise ValueError(
                f"未能从计量模型结果中提取到校准期 scores。"
                f"split_time={self.split_time}, treat_time={self.treat_time}。"
            )

        # 排序并存储，供调试
        n = len(scores)
        self.cal_scores = np.sort(scores)

        # 计算调整分位数 (有限样本覆盖保证)
        # 公式: k = ⌈(n+1)·α⌉, q = sorted_scores[k-1]
        k = int(np.ceil((n + 1) * self.coverage))
        self.quantile = float(self.cal_scores[min(k, n) - 1])

    # =========================================================================
    # 步骤 2: 生成预测区间 (共用 self.train_data 基础)
    # =========================================================================

    def predict(self, post_time_list):
        """用 train_data + post_data 拟合计量模型，生成预测区间。

        数据构成:
          fit_data = self.train_data + post_data
          self.train_data = 训练期 (time < split_time) + 首个处理后时点
          post_data        = 处理后时期 (time > treat_time, 排除已含于 train_data 的首个时点)

        与 fit() 的对应关系:
          fit()      → train_data + cal_data → 校准
          predict()  → train_data + post_data → 预测
          两者共享 train_data 作为共同基础，模型强度对等。

        区间构造:
          CI_t = [effect_t - q,  effect_t + q]
          其中 q 来自 fit() 计算的调整分位数。

        Parameters
        ----------
        post_time_list : list
            处理后时间值列表 (来自 preprocess_data)。

        Returns
        -------
        np.ndarray
            形状 (n_post, 2)，每行为 [lower, upper]。

        Raises
        ------
        ValueError
            若 post_time_list 为空。
        RuntimeError
            若尚未调用 fit() 计算分位数。
        """
        if len(post_time_list) == 0:
            raise ValueError(
                "处理后时间列表为空，无法生成预测区间。"
                "请确保数据中包含处理后时期 (time >= treat_time)。"
            )

        if not hasattr(self, 'quantile'):
            raise RuntimeError(
                "尚未计算分位数，请先调用 fit() 再调用 predict()。"
            )

        # --- 构造预测数据 ---
        # 提取处理后时期数据，排除 treat_time (已包含在 self.train_data 中)
        post_data = self.data[self.data[self.time] > self.treat_time]

        # --- 拟合计量模型 ---
        results = self._fit_model(post_data, caller='predict')

        # --- 提取处理后 effect ---
        post_mask = results[self.time] >= self.treat_time
        post_effects = results.loc[post_mask, 'effect'].to_numpy()

        # --- 构造预测区间 ---
        lower = post_effects - self.quantile
        upper = post_effects + self.quantile

        return np.column_stack((lower, upper))

    # =========================================================================
    # 步骤 3: 格式化为 DataFrame
    # =========================================================================

    def result_to_dataframe(self, confidence_interval, time_list):
        """将预测区间数组转换为 DataFrame。

        与 Full Conformal 的 result_to_dataframe() 输出格式完全一致，
        确保 base.py 的 _merge_results() 能正确合并。

        Parameters
        ----------
        confidence_interval : np.ndarray
            形状 (n, 2) 的数组，每行为 [lower, upper]。
        time_list : list
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

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _fit_model(self, extra_data, caller=''):
        """拼接 train_data + extra_data 并拟合计量模型。

        fit() 和 predict() 共享的拟合逻辑，消除重复的
        concat → fit_econmodel → validate 样板代码。

        Parameters
        ----------
        extra_data : pd.DataFrame
            额外拼接的数据 (fit 时为 cal_data, predict 时为 post_data)。
        caller : str
            调用方标识，用于错误消息。

        Returns
        -------
        pd.DataFrame
            计量模型 fit_econmodel() 的返回值。
        """
        fit_data = pd.concat([self.train_data, extra_data])
        results = self.econ_model.fit_econmodel(
            data=fit_data,
            time=self.time,
            id=self.id,
            y_col=self.y_col,
            treat_col=self.treat_col,
            coverage=self.coverage,
            controls_col=self.controls_col
        )
        self._validate_econ_results(results, caller=caller)
        return results

    def _validate_econ_results(self, results, caller=''):
        """校验计量模型返回的 DataFrame 包含必需的列。

        委托给基类 ConformalBase._validate_econ_results，保留 caller 参数
        以兼容现有调用方式。

        Parameters
        ----------
        results : pd.DataFrame
            计量模型 fit_econmodel() 的返回值。
        caller : str
            调用方标识 ('fit' 或 'predict')，用于错误消息定位。

        Raises
        ------
        TypeError
            若 results 不是 pd.DataFrame。
        ValueError
            若 results 缺少必需的 time 列或 effect 列。
        """
        super()._validate_econ_results(results, context=caller)
