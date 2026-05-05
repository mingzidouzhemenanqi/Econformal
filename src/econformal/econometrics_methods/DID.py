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
            x_cols: List[str] = None, event_window: Tuple[int, int] = None,  **kwargs):
        """
        拟合事件研究模型
        
        参数:
        data: dataframe  包含所有变量的DataFrame
        time: str 时间列名
        id: str 个体ID列名
        y_col: str 因变量列名
        treat_col: str 处理变量列名
        x_cols: list 协变量列名列表
        """
        ########### 数据检查 #############
        # 0. 自动识别event_window，默认窗口期为全样本周期
        if event_window is None:
            self.event_window = self._auto_determine_event_window(data=data, time=time, treat_col=treat_col)
        
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
        if x_cols is not None:
            X = df[event_cols + x_cols]
        else:
            X = df[event_cols]
        # 添加常数项
        X = sm.add_constant(X)

        ########### 回归计算 #############
        # 使用PanelOLS进行面板回归
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
            if t != -1:  # 跳过被省略的基准期
                if t in effect_dict:
                    results_list.append({
                        'relative_time': t,
                        'effect': effect_dict[t]['effect'],
                        'std_error': effect_dict[t]['std_error'],
                        'p-value': effect_dict[t]['p_value'],
                        '置信区间下界': effect_dict[t]['ci_lower'],
                        '置信区间上界': effect_dict[t]['ci_upper']
                    })
            # 对于被省略的基准期，效应为0
            else:
                results_list.append({
                    'relative_time': t,
                    'effect': 0,
                    'std_error': 0,
                    'p-value': 1.0,
                    '置信区间下界': 0,
                    '置信区间上界': 0
                })
        
        self.results_df = pd.DataFrame(results_list)

        # 根据time_mapping和relative_time，将相对处理时间还原成原始时间
        self.results_df[time] = self.results_df['relative_time'].map(time_mapping)
        # 删除relative_time列
        self.results_df = self.results_df.drop(columns=['relative_time'])
        
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
        
        # 标记从未处理的个体
        df['is_treated'] = df[id_col].isin(treated_ids)
        
        # 将从未处理的个体的relative_time设置为NaN
        df.loc[~df['is_treated'], 'relative_time'] = np.nan
        
        # 删除临时列
        df = df.drop(columns=['is_treated'])
        
        return df, time_mapping


    
    def _create_event_dummies(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建事件时间虚拟变量"""
        # 为事件窗口内的每个时间点创建虚拟变量
        min_time, max_time = self.event_window
        
        for t in range(min_time, max_time + 1):
            if t != -1:  # 避免多重共线性，省略一期（通常为t=-1）
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
        # 找到最早的处理时间
        first_treat_time = treat_times.min()
        
        if len(treat_times) == 0:
            raise ValueError("数据中没有处理事件，无法确定事件窗口")
        
        time_list = data[time].unique()

        # 统计time_list中小于first_treat_time的数量
        pre_window = len([t for t in time_list if t < first_treat_time])

        # 统计time_list中大于first_treat_time的数量
        post_window = len([t for t in time_list if t > first_treat_time])
        
        # 确定事件窗口，取前后窗口的最小值以确保所有处理个体都有完整的事件窗口
        event_window = (-pre_window, post_window)
        
        return event_window

