# split_conformal.py
import numpy as np
import pandas as pd
from ..tools import check

class Conformal():
    """
    拆分共形推断实现
    """
    def __init__(self, econ_model, data, time, id, y_col, treat_col, coverage, splite_rate=70, nulls=None, **kwargs):
        self.econ_model = econ_model
        self.coverage = coverage
        self.data = data
        self.time = time
        self.id = id
        self.treat_col = treat_col
        self.y_col = y_col
        self.splite_rate = splite_rate

        # 自动提取处理时间和处理个体（与 Full Conformal 保持一致）
        self.target_id = check.get_treated_individuals(data=data, id=id, time=time, treat_col=treat_col)
        self.treat_time = check.get_first_treatment_year(data=data, id=id, time=time, treat_col=treat_col)


    def compute_conformal_interval(self):
        """共形推断核心行为"""
        """
        完全共形推断流程
        0. 处理数据，从处理前时期切分训练集和校准集，默认切分比例是7:3
        1. 用训练集训练模型，生成残差序列
        2. 计算调整分位数，确定区间范围q
        3. 使用q计算预测区间并返回
        """
        
        '0. 处理数据，从处理前时期切分训练集和校准集'
        # 读取self.time列最小值
        pre_treattime_list = self.data.loc[self.data[self.time] < self.treat_time, self.time].unique()
        # 获取处理前时期70%的样本
        self.pre_treattime_0p7 = np.percentile(pre_treattime_list, self.splite_rate)
        # 
        # 训练集保留self.data中，self.time小于treat_time的行
        #self.data_train = self.data[self.time < self.pre_treattime_0p7]
        # 测试集
        #self.data_cal = self.data[pre_treattime_0p7<=self.time & self.time < treat_time]
        # 处理前
        # self.data_pretreat = self.data[self.time < treat_time]

        '1. 用训练集训练模型，生成残差序列'
        self.fit()

        '3. 使用q计算预测区间并返回'
        self.conformal_interval = self.predict()

        return self.conformal_interval
        
    def fit(self):
        
        data = self.data[self.data[self.time] < self.treat_time]
        residuals = self._get_residuals(data, self.pre_treattime_0p7)

        '2. 计算调整分位数，确定区间范围q'
        n = len(residuals)
        # 计算分位数
        k = int(np.ceil((n + 1) * self.coverage)) 
        # 先排序，后取分位数的值
        self.quantile = np.sort(residuals)[min(k, n - 1)]  # 防止越界
        
        return self
    
    def predict(self):

        # 原始数据去掉校准集
        data = self.data[(self.data[self.time] < self.pre_treattime_0p7) | (self.data[self.time] >= self.treat_time)]
        residuals = self._get_residuals(data, self.treat_time)
        
        lower = residuals - self.quantile
        upper = residuals + self.quantile

        time_list = self.data.loc[self.data[self.time] >= self.treat_time, self.time].unique()

        """将结果转换为DataFrame"""
        return pd.DataFrame({
                f"{int(self.coverage*100)}%_ci_lower": lower,
                f"{int(self.coverage*100)}%_ci_upper": upper},
                index=time_list,  # 使用排序后的不重复时间作为索引
                )


    
    def _get_residuals(self, data, treat_time):

        data_fit = self.econ_model.fit_econmodel(data, self.time, self.id, self.y_col, self.treat_col, self.coverage, train_mode=1)
        data_fit['residuals'] = data_fit['effect']

        # 将['residuals']列中，index大于treat_time的元素提取出
        filtered_data = data_fit.loc[data_fit.index >= treat_time, 'residuals']
        # 转换为 NumPy 向量
        residuals = filtered_data.to_numpy()

        return residuals