import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import cvxpy as cp
import pandas as pd
import numpy as np
from ..tools import check
'''
类核心行为（拟合和预测），使用 fit_econmodel 统一调用
'''

class Econometric(BaseEstimator):

    def __init__(self,):
        # 初始化函数，无参数
        pass

    """
    外部调用接口
        参数:
        data: dataframe  包含所有变量的 DataFrame
        time: str 时间列名
        id: str 个体 ID 列名
        y_col: str 因变量列名
        treat_col: str 处理变量列名
        controls_col: list 协变量列名列表
    """
    def fit_econmodel(self, data: pd.DataFrame, time: str, id: str, y_col: str, treat_col: str, coverage: float,
            controls_col: list = None, **kwargs):

        """
        返回结果：dataframe
            {time}
            effect
            std_error    (处理前残差的标准差)
            p-value      (占位，暂为 0)
            {cov}%_conf_lower
            {cov}%_conf_upper
        """

        ############# 数据预处理 ##############
        '''
        数据预处理：
            0. 识别出处理个体 target_id 和处理时间 treat_time，并检查两者是否唯一（使用 check 中的函数）
            1. 提取控制变量数据（如有），用于后续增强特征矩阵
            2. 将长数据转换为宽数据，行为时间顺序，列为个体顺序
        '''

        # 0. 识别处理个体和处理时间
        self.treat_time, self.target_id = check.extract_treatment_sc(data=data, id=id, time=time, treat_col=treat_col)

        # 1. 提取控制变量数据（在 pivot 之前），用于增强特征矩阵
        controls_data = None
        if controls_col and len(controls_col) > 0:
            controls_data = data[[time, id] + list(controls_col)].copy()

        # 2. 将长数据转换为宽数据，行为时间顺序，列为个体顺序
        data = data[[time, id, y_col]]

        data = data.pivot(index=time, columns=id, values=y_col)
        # pivot 在有缺失 (id, time) 组合时静默产生 NaN，导致后续优化失败
        if data.isna().any().any():
            raise ValueError(
                "数据透视后包含 NaN 值，可能存在不平衡面板。"
                "SC 要求所有个体在所有时点均有观测值（强面板）。"
                "请检查数据中是否每个 (id, time) 组合均有且仅有一条记录。"
            )

        ############# 控制变量增强 ##############
        # 使用控制变量的处理前均值增强特征矩阵，使权重优化同时匹配结果变量和协变量
        self._has_controls = False
        if controls_data is not None and len(controls_col) > 0:
            y_target = data[self.target_id]
            X_controls = data.drop(columns=[self.target_id])
            pre_mask = data.index < self.treat_time

            aug_rows_y = []
            aug_rows_X = []
            for c in controls_col:
                c_wide = controls_data.pivot(index=time, columns=id, values=c)
                c_pre = c_wide[c_wide.index < self.treat_time]
                c_means = c_pre.mean(axis=0)  # Series: index=unit_id
                # 标准化以与结果变量行处于可比尺度
                c_std = c_means.std(ddof=1)
                if c_std < 1e-10:
                    c_std = 1.0  # 常数列退化为等权贡献
                c_means_std = (c_means - c_means.mean()) / c_std
                aug_rows_y.append(c_means_std[self.target_id])
                aug_rows_X.append(c_means_std[X_controls.columns].values)

            # 边界检查：控制变量数不应过多主导优化
            n_pre = len(y_target[pre_mask])
            if len(aug_rows_y) > n_pre:
                warnings.warn(
                    f"控制变量数 ({len(aug_rows_y)}) 超过处理前时期数 ({n_pre})。"
                    f"协变量匹配可能主导优化目标，建议减少控制变量数量。"
                )

            self._y_aug = np.concatenate([y_target[pre_mask].values, np.array(aug_rows_y)])
            self._X_aug = np.vstack([X_controls[pre_mask].values] + aug_rows_X)
            self._has_controls = True

        ############# 模型拟合 ##############
        """
        模型拟合
            0. 模型拟合：仅使用处理前数据拟合权重（如有控制变量则使用增强矩阵）
            1. 计算预测值：使用拟合的权重计算所有个体的预测值
            2. 计算处理效应：预测值与实际值之差
            3. 计算 result_df 各列数据
        """
        # SC 方法必须仅使用处理前数据拟合权重（这是合成控制法的基本原理）
        # Full Conformal 兼容：当仅 1 个处理后时期时，该时期（已被 null 偏移）需参与
        # 权重学习。这遵循 CWZ (2021) 的方法论：模型应在原假设下重新估计。
        self._full_conformal_mode = len(data[data.index >= self.treat_time]) == 1
        if self._full_conformal_mode:
            if self._has_controls:
                post_mask = data.index >= self.treat_time
                y_post = data[self.target_id][post_mask].values
                X_post = data.drop(columns=[self.target_id])[post_mask].values
                self.fit(np.vstack([self._X_aug, X_post]),
                         np.concatenate([self._y_aug, y_post]))
            else:
                self.fit(data.drop(columns=[self.target_id]).values, data[self.target_id].values)
        else:
            if self._has_controls:
                self.fit(self._X_aug, self._y_aug)
            else:
                pre_treat_data = data[data.index < self.treat_time]
                self.fit(pre_treat_data.drop(columns=[self.target_id]).values, pre_treat_data[self.target_id].values)

        # 当使用控制变量时，fit() 存储的 pre_treat_residuals_ 包含混合尺度的残差
        # （结果变量行 + 标准化控制变量行 + 可能的处理后行），切除控制变量行，
        # 仅保留结果变量部分以确保 std_error 和传统 CI 基于正确的残差计算
        if self._has_controls:
            n_outcome_rows = len(data[data.index < self.treat_time])
            ctrl_slice = slice(n_outcome_rows, n_outcome_rows + len(controls_col))
            self.pre_treat_residuals_ = np.delete(self.pre_treat_residuals_, ctrl_slice)

        # Full Conformal 模式下，fit() 包含了处理后数据（已被 null 偏移）。
        # 切除最后一个残差（处理后行），使 std_error / CI 仅基于处理前残差，
        # 避免处理效应（和 null 偏移）对标准误估计的污染。
        if self._full_conformal_mode:
            self.pre_treat_residuals_ = self.pre_treat_residuals_[:-1]

        # 1. 计算 result_df 各列数据
        
        # 获取拟合结果，组成 dataframe
        """
            {time}         时间
            effect         SC 预测值与实际值之差
            std_error      处理前残差的标准差
            p-value        占位，暂为 0
            {cov}%_conf_lower  传统 CI 下界
            {cov}%_conf_upper  传统 CI 上界
        """
        data['predictions'] = self.predict(data.drop(columns=[self.target_id]).values)

        data['effect'] = self.get_residuals(data)

        data['std_error'] = self.get_std_error()

        data['p-value'] = self.get_p_value()

        half_lo, half_hi = self.get_confidence_interval(coverage)
        ci_lower_col = f"{int(coverage * 100)}%_conf_lower"
        ci_upper_col = f"{int(coverage * 100)}%_conf_upper"
        # CI centered on each period's effect: [effect + half_lo, effect + half_hi]
        data[ci_lower_col] = data['effect'] + half_lo
        data[ci_upper_col] = data['effect'] + half_hi

        # 2. 数据表处理
        # 直接将 index（时间）作为列加入并构造干净的 DataFrame，
        # 避免 .values 的 dtype 统一化导致时间类型丢失。
        data[time] = data.index
        target_columns = [time, 'effect', 'std_error', 'p-value', ci_lower_col, ci_upper_col]
        result_df = data[target_columns].reset_index(drop=True)
        # 确保时间列保持原始 dtype，不强制转换
        # 将 time 列设置为 index
        # result_df.set_index(time, inplace=True)

        return result_df

    def fit(self, X, y):
        '''
        X: 用于拟合被解释变量的其他变量
        y: 被解释变量
        '''

        # 检查输入的 X 和 y 是否满足要求
        X, y = check_X_y(X, y)
    
        # 定义变量 w，其维度为 X 的列数
        w = cp.Variable(X.shape[1])
        # 定义目标函数，最小化 X@w - y 的平方和
        objective = cp.Minimize(cp.sum_squares(X@w - y))
        
        # 定义约束条件，w 的和为 1，w 的每个元素都大于等于 0
        constraints = [cp.sum(w) == 1, w >= 0]
        
        # 定义问题，包含目标函数和约束条件
        problem = cp.Problem(objective, constraints)
        # 求解问题
        problem.solve(verbose=False)
        
        # 检查求解状态
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"SC 权重优化求解失败，求解器状态：{problem.status}")

        if problem.status == 'optimal_inaccurate':
            warnings.warn(
                f"SC 权重优化返回 'optimal_inaccurate'，解可能数值不精确。"
                f"建议检查数据缩放或考虑标准化变量。"
            )

        # 将 X，y，w 的值赋给实例变量
        self.X_ = X
        self.y_ = y
        self.w_ = w.value
        
        # 保存处理前残差用于计算标准误和置信区间
        self.pre_treat_residuals_ = y - X @ self.w_
        
        # 设置 is_fitted_ 为 True，表示模型已经拟合
        self.is_fitted_ = True
        return self
        
        
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        return X @ self.w_
    
    # 计算残差
    def get_residuals(self,data):
        # 此处获取的残差，是全样本训练后的残差
        return data[self.target_id] - data['predictions']

    def get_std_error(self):
        check_is_fitted(self)
        if len(self.pre_treat_residuals_) < 2:
            return float('nan')
        return np.std(self.pre_treat_residuals_, ddof=1)

    def get_p_value(self):
        check_is_fitted(self)
        return float('nan')

    def get_confidence_interval(self, coverage=0.9):
        """返回基于正态近似的置信区间半宽 (z * std_err)。

        注意：返回的是半宽而非完整区间，由调用方与 effect 组合为
        [effect - half_width, effect + half_width]。
        """
        check_is_fitted(self)
        std_err = np.std(self.pre_treat_residuals_, ddof=1)
        if np.isnan(std_err) or std_err < 1e-15:
            return float('nan'), float('nan')
        from scipy import stats
        z = stats.norm.ppf((1 + coverage) / 2)
        half_width = z * std_err
        return -half_width, half_width
