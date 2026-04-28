from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import cvxpy as cp
import pandas as pd
import numpy as np
from ..tools import check, plot
'''
类核心行为（拟合和预测），使用fit_econmodel统一调用
'''

class Econometric():

    def __init__(self,):
        # 初始化函数，无参数
        pass

    """
    外部调用接口
    输入参数: data: dataframe  数据表
            time: str    时间列名
            id: str      个体列名
            y_col: str   因变量列名
            train_mode: int  0:仅使用训练集训练数据；1:使用所有样本数据训练。默认为1
                    train_mode存在是为了适应SC等类机器学习的。这类方法只需要使用处理前数据进行训练。但是在进行共形推断以及DID等方法回归时，需要使用所有数据进行训练
    """
    def fit_econmodel(self, data: pd.DataFrame, time: str, id: str, y_col: str, treat_col: str, train_mode: int=1):

        """
        返回结果：dataframe
            
            year
            prediction   如果是DID这种直接出effcet的方法，为0
            effect
            std_error    暂时为0
            p-value      暂时为0
            置信区间下界  暂时为0
            置信区间上界  暂时为0
        """

        ############# 数据预处理 ##############
        '''
        数据预处理：
            0. 识别出处理个体target_id和处理时间treat_time，并检查两者是否唯一（使用check中的函数）
            1. 去除控制变量，合成控制法暂时无法处理带有控制变量情况
            2. 将长数据转换为宽数据，行为时间顺序，列为个体顺序
        '''
        # 0. 去除控制变量，数据集data仅保留time,id,y_col列，去除协变量
        self.treat_time, self.target_id = check.extract_treatment_sc(data=data, id=id, time=time, treat_col=treat_col)
        
        # 1. 识别出处理个体target_id和处理时间treat_time，并检查两者是否唯一
        data = data[[time, id, y_col]]


        # 将长数据转换为宽数据，行为时间顺序，列为个体顺序
        data = data.pivot(index=time, columns=id, values=y_col)

        ############# 模型拟合 ##############
        """
        模型拟合
            0. 指定拟合模式：train_mode模型训练模式：0:仅使用训练集训练数据；1:使用所有样本数据训练。默认为1
                train_mode存在是为了适应SC等类机器学习的。这类方法只需要使用处理前数据进行训练。但是在进行共形推断以及DID等方法回归时，需要使用所有数据进行训练
            1. 计算result_df各列数据
            2. 数据表处理
        """
        # 0. train_mode模型训练模式：0:仅使用训练集训练数据；1:使用所有样本数据训练。默认为1
        if train_mode == 0:
            train = data[data.index < self.treat_time]
            self.fit(train.drop(columns=[self.target_id]), train[self.target_id])
        else:
            self.fit(data.drop(columns=[self.target_id]), data[self.target_id])
        
        # 1. 计算result_df各列数据
        
        # 获取拟合结果，组成dataframe
            """
                year         时间
                prediction   SC的预测值
                effect       SC预测值与实际值之差
                std_error    暂时为0
                p-value      暂时为0
                置信区间下界  暂时为0
                置信区间上界  暂时为0
            """
        data['predictons'] = self.predict(data.drop(columns=[self.target_id]))

        data['effect'] = self.get_residuals(data)

        data['std_error'] = self.get_std_error()

        data['p-value'] = self.get_p_value()

        data['置信区间下界'] = self.get_confidence_interval()

        data['置信区间上界'] = self.get_confidence_interval()

        # 2. 数据表处理
        #由于data的index存在多级索引问题，直接data.reset_index()无法去除列名，故直接提取data所需数据，组成新的dataframe
        #将data的index变成一列，列名为time
        data[time] = data.index
        # 保留所需列
        target_columns = [time, 'predictons', 'effect', 'std_error', 'p-value', '置信区间下界', '置信区间上界']
        data = data[target_columns]
        # 提取纯净数据
        clean_data = data.values
        result_df = pd.DataFrame(
            data=clean_data,
            columns=target_columns  # 使用原始列名
        )
        # 将time列的数值转化为int
        result_df[time] = result_df[time].astype(int)
        # 将time列设置为index
        # result_df.set_index(time, inplace=True)

        return result_df

    def fit(self, X, y):
        '''
        X: 用于拟合被解释变量的其他变量
        y: 被解释变量
        '''

        # 传入数据为常规面板数据，需要将长数据转换为宽数据
        

        # 检查输入的X和y是否满足要求
        X, y = check_X_y(X, y)
    
        # 定义变量w，其维度为X的列数
        w = cp.Variable(X.shape[1])
        # 定义目标函数，最小化X@w - y的平方和
        objective = cp.Minimize(cp.sum_squares(X@w - y))
        
        # 定义约束条件，w的和为1，w的每个元素都大于等于0
        constraints = [cp.sum(w) == 1, w >= 0]
        
        # 定义问题，包含目标函数和约束条件
        problem = cp.Problem(objective, constraints)
        # 求解问题
        problem.solve(verbose=False)
        
        # 将X，y，w的值赋给实例变量
        self.X_ = X
        self.y_ = y
        self.w_ = w.value
        
        # 设置is_fitted_为True，表示模型已经拟合
        self.is_fitted_ = True
        return self
        
        
    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)
        
        return X @ self.w_
    
    # 计算残差
    def get_residuals(self,data):
        # 此处获取的残差，是全样本训练后的残差

        data['effect'] = data[self.target_id] - data['predictons']

        # 返回self.result_df['effect']这一列，不要变成Series格式

        return data[['effect']]

    def get_std_error(self):

        return 0
    
    def get_p_value(self):

        return 0

    def get_confidence_interval(self):

        return 0