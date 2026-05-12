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
            
            year
            prediction   如果是 DID 这种直接出 effcet 的方法，为 0
            effect
            std_error    暂时为 0
            p-value      暂时为 0
            置信区间下界  暂时为 0
            置信区间上界  暂时为 0
        """

        ############# 数据预处理 ##############
        '''
        数据预处理：
            0. 识别出处理个体 target_id 和处理时间 treat_time，并检查两者是否唯一（使用 check 中的函数）
            1. 去除控制变量，合成控制法暂时无法处理带有控制变量情况
            2. 将长数据转换为宽数据，行为时间顺序，列为个体顺序
        '''

        # 0. 去除控制变量，数据集 data 仅保留 time,id,y_col 列，去除协变量
        self.treat_time, self.target_id = check.extract_treatment_sc(data=data, id=id, time=time, treat_col=treat_col)
        
        # 1. 识别出处理个体 target_id 和处理时间 treat_time，并检查两者是否唯一
        data = data[[time, id, y_col]]


        # 将长数据转换为宽数据，行为时间顺序，列为个体顺序
        data = data.pivot(index=time, columns=id, values=y_col)

        ############# 模型拟合 ##############
        """
        模型拟合
            0. 模型拟合：仅使用处理前数据拟合权重
            1. 计算预测值：使用处理前数据拟合的权重，计算所有个体的预测值
            2. 计算处理效应：预测值与实际值之差
            3. 计算 result_df 各列数据
        """
        # SC 方法必须仅使用处理前数据拟合权重（这是合成控制法的基本原理）
        # 对full模式进行兼容：如果处理后期数只有1期数据，保留所有数据。如果处理后有多个期数据，则仅保留处理前数据
        if len(data[data.index >= self.treat_time]) == 1:   ## 对full模式进行兼容,如果处理后期数只有1期数据，使用全样本训练
            self.fit(data.drop(columns=[self.target_id]).values, data[self.target_id].values)          
        else:
            pre_treat_data = data[data.index < self.treat_time]
            self.fit(pre_treat_data.drop(columns=[self.target_id]).values, pre_treat_data[self.target_id].values)
        
        # 1. 计算 result_df 各列数据
        
        # 获取拟合结果，组成 dataframe
        """
            year         时间
            prediction   SC 的预测值
            effect       SC 预测值与实际值之差
            std_error    暂时为 0
            p-value      暂时为 0
            置信区间下界  暂时为 0
            置信区间上界  暂时为 0
        """
        data['predictions'] = self.predict(data.drop(columns=[self.target_id]).values)

        data['effect'] = self.get_residuals(data)

        data['std_error'] = self.get_std_error()

        data['p-value'] = self.get_p_value()

        ci_lower, ci_upper = self.get_confidence_interval(coverage)
        data['置信区间下界'] = ci_lower
        data['置信区间上界'] = ci_upper

        # 2. 数据表处理
        #由于 data 的 index 存在多级索引问题，直接 data.reset_index() 无法去除列名，故直接提取 data 所需数据，组成新的 dataframe
        #将 data 的 index 变成一列，列名为 time
        data[time] = data.index
        # 保留所需列
        target_columns = [time, 'predictions', 'effect', 'std_error', 'p-value', '置信区间下界', '置信区间上界']
        data = data[target_columns]
        # 提取纯净数据
        clean_data = data.values
        result_df = pd.DataFrame(
            data=clean_data,
            columns=target_columns  # 使用原始列名
        )
        # 将 time 列的数值转化为 int
        result_df[time] = result_df[time].astype(int)
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
        if not hasattr(self, 'pre_treat_residuals_'):
            return 0
        return np.std(self.pre_treat_residuals_, ddof=1)
    
    def get_p_value(self):

        return 0

    def get_confidence_interval(self, coverage=0.9):
        if not hasattr(self, 'pre_treat_residuals_'):
            return 0, 0
        std_err = np.std(self.pre_treat_residuals_, ddof=1)
        from scipy import stats
        z = stats.norm.ppf((1 + coverage) / 2)
        return -z * std_err, z * std_err
