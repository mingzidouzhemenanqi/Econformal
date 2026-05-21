import pandas as pd
import numpy as np
from tqdm import tqdm
from .conformal_base import ConformalBase

class Conformal(ConformalBase):
    """
    完全共形推断实现

    """
    def __init__(self, econ_model, data: pd.DataFrame, time: str, id: str,
                y_col: str, treat_col: str, coverage: float, 
                nulls: list,
                controls_col: list=[], **kwargs):
        self.nulls = nulls
        
        # 调用父类 __init__，复用所有公共初始化逻辑
        super().__init__(
            econ_model=econ_model, data=data,time=time,id=id,
            y_col=y_col,treat_col=treat_col,
            coverage=coverage, controls_col=controls_col,
            **kwargs
        )


    def compute_conformal_interval(self):
        """
        完全共形推断流程
        0. 处理数据，并预先形成所需变量
        1. 共形推断主逻辑，计算p_value_matrix
        2. 根据p_value_matrix和nulls，计算置信区间
        3. 处理成dataframe，返回结果
        """
        
        '0. 处理数据，并预先形成所需要的变量'
        # time_list 时间列表，未处理时间
        # p_value_matrix p值空矩阵，行数为未处理时间长度，列数为nulls的长度
        time_list, p_value_matrix = self.preprocess_data()

        '1. 共形推断主逻辑，计算p_value_matrix'
        self.p_value_matrix = self.fit(time_list, p_value_matrix)

        '2. 用p_value_matrix从nulls中找出上下界，组成置信区间'
        confidence_interval = self.predict()

        '3. 结果转化，将矩阵转化成dataframe'
        self.conformal_interval = self.result_to_dataframe(confidence_interval, time_list)

        return self.conformal_interval
    

    def fit(self, time_list, p_value_matrix):
        """
        完全共形推断计算流程
        """

        '第一层循环：时间'
        #for i, treat_time_conformal in tqdm(enumerate(time_list), desc='p_value_matrix计算: '):
        for i, treat_time_conformal in enumerate(tqdm(time_list, desc='p_value_matrix计算: ')):
            
            # 提取self.data中，self.time列的小于self.treat_time的行,且self.time等于treat_time_conformal的行，作为准增强数据（y_new在nulls的循环中修改即可）
            conformal_data_pro_0 = self.data[(self.data[self.time] < self.treat_time) | (self.data[self.time] == treat_time_conformal)]
            
            '第二层循环，遍历nulls对应行的每一个null'
            # 提前生成定长数组，提高运行速度
            p_value_list = np.zeros(len(self.nulls))
            for j, null in enumerate(self.nulls):
                # 采用copy的方式，让conformal_data_pro每次循环重置，不然会一直累加null
                conformal_data_pro = conformal_data_pro_0.copy()
                
                # 将conformal_data_pro中位于self.y_col列，同时self.id在列表self.target_id_list中且self.time等于treat_time_conformal处的值，减去null。得到增强数据集 
                conformal_data_pro.loc[conformal_data_pro[self.id].isin(self.target_id_list) & (conformal_data_pro[self.time] == treat_time_conformal), self.y_col] -= null
                
                # 使用增强数据训练模型            
                fit_econmodel_data = self.econ_model.fit_econmodel(data=conformal_data_pro, time=self.time, id=self.id, y_col=self.y_col, treat_col=self.treat_col, coverage=self.coverage)

                # 获取处理效应
                residuals = fit_econmodel_data[['effect']]
                #print(residuals)

                # 根据残差计算P值, 并存储在向量对应位置
                p_value_list[j] = self._get_pvalue_each_permutations(residuals)

            # 单周期循环完成，按行存储结果
            p_value_matrix[i] = p_value_list

        return p_value_matrix
    
    def predict(self):
        """为每个新样本计算预测区间"""
        # 生成nulls矩阵,（行为p_value_matrix行数，列为nulls长度，每行都是nulls，方便找区间）
        nulls_matrix = np.tile(self.nulls, (self.p_value_matrix.shape[0], 1))

        # 找到self.p_value_matrix中p值小于1-coverage的位置，并将对应nulls设为NaN
        mask = self.p_value_matrix < (1-self.coverage)

        nulls_matrix[mask] = np.nan

        # 计算nulls_matrix每行的最小值和最大值（忽略NaN）
        min_values = np.nanmin(nulls_matrix, axis=1)
        max_values = np.nanmax(nulls_matrix, axis=1)
        
        # 组合为两列数组
        confidence_interval = np.column_stack((min_values, max_values))

        return confidence_interval

    def preprocess_data(self):

        # 获取第一层循环变量：时间
        # 获取self.data中self.time列所有大于等于self.treat_time的值，返回一个列表
        time_list = self.data.loc[self.data[self.time] >= self.treat_time, self.time].unique()

        # p值空矩阵，用于存储每个null的计算结果，提前生成矩阵提高运算速度。行数为处理后时期长度，列为nulls的长度       
        p_value_matrix = np.zeros((len(time_list), len(self.nulls)))

        return time_list, p_value_matrix

    def result_to_dataframe(self, confidence_interval, time_list):
        """将结果转换为DataFrame"""
        return pd.DataFrame(
                        confidence_interval,
                        index=time_list,  # 使用排序后的不重复时间作为索引
                        columns=[
                            f"{int((self.coverage)*100)}%_conformal_lower",
                            f"{int((self.coverage)*100)}%_conformal_upper"
                        ])

    def _get_pvalue_each(self, resid_df):
        # 将resid_df转成numpy数组
        residuals = resid_df.to_numpy().flatten()
        
        # residuals中的值取绝对值
        residuals = np.abs(residuals)
        
        # 计算每个残差小于最后一个残差的比例
        p_values = np.mean(residuals <= residuals[-1])
        
        return p_values
    def _get_pvalue_each_permutations(self, resid_df):
        # residual为一列的dataframe，转成numpy数组
        #residuals = residuals.to_numpy().flatten()
        # 残差小于最后一个残差的比例
        #return np.mean(residuals <= residuals[-1])
        # 给resid_df生成新列"post_intervention"，除了最后一个值为True，其他都是False
        resid_df['post_intervention'] = False  # 先将所有值设为False
        # 将'post_intervention'列最后一个元素设置为True
        resid_df.iloc[-1, resid_df.columns.get_loc('post_intervention')] = True
        
        u = resid_df["effect"].values
        post_intervention = resid_df["post_intervention"].values
        
        block_permutations = np.stack([np.roll(u, permutation, axis=0)[post_intervention]
                                    for permutation in range(len(u))])
        
        statistics = self._test_statistic(block_permutations, q=1, axis=1)
        
        p_val = np.mean(statistics >= statistics[0])

        return p_val

    def _test_statistic(self, u_hat, q=1, axis=0):
        return (np.abs(u_hat) ** q).mean(axis=axis) ** (1/q)
    
