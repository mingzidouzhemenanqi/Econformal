import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
# from sklearn.base import BaseEstimator, RegressorMixin
from typing import List, Tuple
'''
类核心行为（拟合和预测），使用fit_econmodel统一调用
'''

class Econometric():
    def __init__(self):
        """
        初始化事件研究法模型
        """
        pass
    
    def fit_econmodel(self, data: pd.DataFrame, time: str, id: str, y_col: str, treat_col: str, coverage: float,
            controls_col: List[str] = None, event_window: Tuple[int, int] = None,  **kwargs):
        """
        拟合事件研究模型
        
        参数:
        data: dataframe  包含所有变量的DataFrame
        time: str 时间列名
        id: str 个体ID列名
        y_col: str 因变量列名
        treat_col: str 处理变量列名
        coverage: float 覆盖率

        controls_col: list 协变量列名列表
        event_window: tuple 事件窗口，默认None
        """
        """
        返回数据格式：dataframe
                year
                prediction   
                effect
                std_error    
                p-value      
                90%_conf_lower
                90%_conf_upper
        """
        ci_lower_col = f"{int(coverage * 100)}%_conf_lower"
        ci_upper_col = f"{int(coverage * 100)}%_conf_upper"
        ########### 数据检查 #############
        # 0. 自动识别event_window，默认窗口期为全样本周期。
        # 注意：event_window 不缓存。每次调用都会基于当前数据重新计算，
        # 因为在 conformal 子采样中，数据的相对时间映射可能不同。
        # 若用户显式传入 event_window，F1 的 kwargs 透传机制确保
        # 所有内外拟合使用相同窗口。
        if event_window is None:
            self.event_window = self._auto_determine_event_window(data=data, time=time, treat_col=treat_col)
        else:
            self.event_window = event_window
        
        # 1. 检查data中变量列名是否有以‘event_’开头的变量，如果有则抛出错误，并提示修改变量名，否则会导致估计出错
        if any(col.startswith('event_') for col in data.columns):
            raise ValueError("变量名不能以'event_开头'，否则会导致估计出错，请修改变量名。")

        ########### 数据预处理 #############
        # 1. 复制数据以避免修改原数据
        df = data.copy()
        # 2. 计算相对事件时间，同时保存相对时间与原始时间的映射
        df, time_mapping = self._calculate_relative_time(df, time, treat_col, id)
        # 3. 创建事件时间虚拟变量
        df = self._create_event_dummies(df)


        # 4. 准备回归数据
        df = df.set_index([id, time])  # 设置面板数据
        event_cols = [col for col in df.columns if col.startswith('event_')]  # 事件时间虚拟变量列名
        y = df[y_col] # 因变量
        # 准备自变量：事件虚拟变量 + 协变量
        if controls_col is not None:
            X = df[event_cols + controls_col]
        else:
            X = df[event_cols]
        # 添加常数项
        X = sm.add_constant(X)

        ########### 回归计算 #############
        # 过滤掉全零 event dummies（conformal 子采样数据中某些相对时间可能无观测）
        nonzero_event_cols = [c for c in event_cols if X[c].abs().sum() > 1e-12]
        if not nonzero_event_cols:
            raise ValueError(
                "所有事件虚拟变量均为全零列，无法进行 DID 估计。"
                "可能原因：event_window 中的相对时间在子采样数据中不存在。"
                "请检查 event_window 是否超出数据的时间范围，或尝试让模型自动检测窗口。"
            )
        # 重建 X 矩阵（仅保留非全零的 event dummies + controls + constant）
        if controls_col is not None:
            X = df[nonzero_event_cols + controls_col]
        else:
            X = df[nonzero_event_cols]
        X = sm.add_constant(X)
        event_cols = nonzero_event_cols  # 后续提取结果仅遍历存在的 event dummies

        # 使用 PanelOLS 进行面板回归。
        # 注意：entity_effects=True 和 time_effects=True 与事件时间虚拟变量
        # （event dummies）存在隐含共线性——时间固定效应是所有 event dummies 的
        # 线性组合。PanelOLS 会通过内部降秩处理静默解决，但升维后的系数解释
        # 与传统不含时间 FE 的事件研究法略有不同。这是计量设计选择，非错误。
        self.model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        self.results = self.model.fit(cov_type='clustered', cluster_entity=True)
        

        ########### 提取结果 #############
        # 提取事件系数
        effect_dict = {}
        for col in event_cols:
            t = int(col.split('_')[1])
            effect_dict[t] = {
                'effect': self.results.params[col],
                'std_error': self.results.std_errors[col],
                'p_value': self.results.pvalues[col],
                'ci_lower': self.results.conf_int(level=coverage).loc[col, 'lower'],
                'ci_upper': self.results.conf_int(level=coverage).loc[col, 'upper']
            }
        
        # 创建结果DataFrame
        results_list = []
        for t in range(self.event_window[0], self.event_window[1] + 1):
            if t != self._omitted:  # 跳过基准期
                if t in effect_dict:
                    results_list.append({
                        'relative_time': t,
                        'effect': effect_dict[t]['effect'],
                        'std_error': effect_dict[t]['std_error'],
                        'p-value': effect_dict[t]['p_value'],
                        ci_lower_col: effect_dict[t]['ci_lower'],
                        ci_upper_col: effect_dict[t]['ci_upper']
                    })
                else:
                    # 该相对时间的 event dummy 为全零列（子采样数据中不存在），
                    # 无法估计，填充 NaN 而非静默跳过
                    results_list.append({
                        'relative_time': t,
                        'effect': float('nan'),
                        'std_error': float('nan'),
                        'p-value': float('nan'),
                        ci_lower_col: float('nan'),
                        ci_upper_col: float('nan')
                    })
            # 对于被省略的基准期，效应为0
            else:
                results_list.append({
                    'relative_time': t,
                    'effect': 0,
                    'std_error': 0,
                    'p-value': 1.0,
                    ci_lower_col: 0,
                    ci_upper_col: 0
                })
        
        self.results_df = pd.DataFrame(results_list)

        # 根据time_mapping和relative_time，将相对处理时间还原成原始时间
        self.results_df[time] = self.results_df['relative_time'].map(time_mapping)
        # 删除relative_time列
        self.results_df = self.results_df.drop(columns=['relative_time'])
        # 过滤掉时间映射失败的 rows（event_window 超出数据范围时会产生 NaN 时间）
        n_before = len(self.results_df)
        self.results_df = self.results_df.dropna(subset=[time])
        n_dropped = n_before - len(self.results_df)
        if n_dropped > 0:
            import warnings
            warnings.warn(
                f"event_window 中的 {n_dropped} 个相对时间在数据中无对应绝对时间，"
                f"已从结果中移除。请检查 event_window 是否超出数据的时间范围。"
            )
        
        return self.results_df

    def _calculate_relative_time(self, data: pd.DataFrame, time_col: str, treat_col: str, id_col: str) -> Tuple[pd.DataFrame, dict]:
        """
        生成相对处理时间并创建映射字典，将从未处理的样本的relative_time设置为NaN
        
        参数:
        data: 输入数据
        time_col: 时间列名
        treat_col: 处理变量列名
        id_col: 个体ID列名
        
        返回:
        Tuple: (包含相对时间列的DataFrame, 映射字典{相对时间: 原始时间})
        """
        # 复制数据以避免修改原数据
        df = data.copy()
        
        # 1. 获取所有唯一的时间点并排序
        all_times = sorted(df[time_col].unique())
        
        # 2. 确定处理时间点
        treat_times = df[df[treat_col] == 1][time_col].unique()
        
        if len(treat_times) == 0:
            raise ValueError("数据中没有处理事件")

        # DID 当前仅支持所有处理个体在同一时间首次被处理。
        # 检查每个个体的首次处理时间是否一致。
        first_treat = df[df[treat_col] == 1].groupby(id_col)[time_col].min()
        unique_first_treat = first_treat.unique()
        if len(unique_first_treat) > 1:
            raise ValueError(
                f"检测到多个不同的首次处理时间: {sorted(unique_first_treat)}。"
                f"DID 当前仅支持所有处理个体在同一时点首次接受处理（非交错处理）。"
                f"请使用 SDID 或其他支持交错处理的方法。"
            )

        # 假设所有样本在同一时间被处理
        treat_time = treat_times.min()
        
        # 3. 找到处理时间点在时间列表中的位置
        if treat_time not in all_times:
            raise ValueError(f"处理时间{treat_time}不在数据的时间点中")
        
        treat_index = all_times.index(treat_time)
        
        # 4. 创建相对时间到原始时间的映射字典
        time_mapping = {}
        
        # 处理前时期映射
        for i in range(treat_index):
            relative_time = i - treat_index  # 负值
            time_mapping[relative_time] = all_times[i]
        
        # 处理当期映射
        time_mapping[0] = all_times[treat_index]
        
        # 处理后时期映射
        for i in range(treat_index + 1, len(all_times)):
            relative_time = i - treat_index  # 正值
            time_mapping[relative_time] = all_times[i]
        
        # 5. 在数据中添加相对时间列
        # 创建原始时间到相对时间的反向映射
        reverse_mapping = {v: k for k, v in time_mapping.items()}
        
        # 添加相对时间列
        df['relative_time'] = df[time_col].map(reverse_mapping)
        
        # 6. 识别从未处理的样本，并将其relative_time设置为NaN
        # 找到所有被处理的个体
        treated_ids = df[df[treat_col] == 1][id_col].unique()

        # 检查是否存在对照组：若所有个体均被处理，事件虚拟变量与时间固定效应完全共线
        all_ids = df[id_col].unique()
        if len(treated_ids) == len(all_ids):
            raise ValueError(
                f"所有 {len(all_ids)} 个个体均为处理组，没有对照组。"
                f"DID 事件研究法需要至少一个从未被处理的个体作为控制组，"
                f"以识别事件虚拟变量与时间固定效应。"
            )

        # 标记从未处理的个体
        df['is_treated'] = df[id_col].isin(treated_ids)
        
        # 将从未处理的个体的relative_time设置为NaN
        df.loc[~df['is_treated'], 'relative_time'] = np.nan
        
        # 删除临时列
        df = df.drop(columns=['is_treated'])
        
        return df, time_mapping


    
    def _create_event_dummies(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建事件时间虚拟变量

        注意：硬编码省略 t=-1（处理前最后一期）作为基准期以避免多重共线性。
        若 event_window 不包含 -1（如窗口为 (0, post_window)），则没有虚拟变量
        被省略，所有 event dummies + time_effects + entity_effects 将完全共线。
        此时 PanelOLS 会静默丢弃变量。应确保 event_window 包含 t=-1。
        """
        # 为事件窗口内的每个时间点创建虚拟变量
        min_time, max_time = self.event_window

        # 确定要省略的基准期：优先 t=-1，若不在窗口中则选最小相对时间
        self._omitted = -1 if min_time <= -1 <= max_time else min_time

        for t in range(min_time, max_time + 1):
            if t != self._omitted:  # 避免多重共线性，省略基准期
                data[f'event_{t}'] = (data['relative_time'] == t).astype(int)
        
        return data
    
    def _auto_determine_event_window(self, data: pd.DataFrame, time: str, treat_col: str) -> Tuple[int, int]:
        """
        自动确定事件窗口范围
        
        参数:
        data: 输入数据
        time: 时间列名
        treat_col: 处理变量列名
        
        # 计算整个样本的时间范围
        min_time = data[time].min()
        max_time = data[time].max()
        
        # 计算事件前最大可能窗口
        pre_window = -(first_treat_time - min_time)
        返回:
        事件窗口范围 (min_time, max_time)
        """
        # 找到所有处理事件的时间点
        treat_times = data[data[treat_col] == 1][time].unique()

        if len(treat_times) == 0:
            raise ValueError("数据中没有处理事件，无法确定事件窗口")

        # 找到最早的处理时间
        first_treat_time = treat_times.min()
        
        time_list = data[time].unique()

        # 统计time_list中小于first_treat_time的数量
        pre_window = len([t for t in time_list if t < first_treat_time])

        # 统计time_list中大于first_treat_time的数量
        post_window = len([t for t in time_list if t > first_treat_time])
        
        # 确定事件窗口，取前后窗口的最小值以确保所有处理个体都有完整的事件窗口
        event_window = (-pre_window, post_window)
        
        return event_window

