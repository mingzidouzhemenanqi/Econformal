import sys

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
                controls_col: list=None, **kwargs):
        if controls_col is None:
            controls_col = []
        if nulls is None or len(nulls) == 0:
            raise ValueError(
                "Full Conformal 方法需要用户提供 nulls 参数（原假设列表）。"
                "例如: nulls=np.linspace(-10, 10, 100)"
            )
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

        # 检查 p_value_matrix 是否所有原假设均被拒绝（nulls 范围太窄）
        all_rejected = np.all(self.p_value_matrix < (1 - self.coverage), axis=1)
        if all_rejected.any():
            affected = time_list[all_rejected]
            print(
                f"\n[WARNING] 在时间点 {list(affected)}，所有原假设均被拒绝（alpha={1-self.coverage:.2f}）。"
                f"当前 nulls 范围 [{self.nulls[0]:.4f}, {self.nulls[-1]:.4f}] 可能太窄，"
                f"建议扩大 nulls 范围（如 np.linspace(-50, 50, 100)）以确保能覆盖置信区间边界。\n",
                file=sys.stderr,
                flush=True
            )

        '2. 用p_value_matrix从nulls中找出上下界，组成置信区间'
        confidence_interval = self.predict()

        # 检查是否所有置信区间边界均为 NaN（nulls 范围完全不够）
        nan_mask = np.isnan(confidence_interval).all(axis=1)
        if nan_mask.any():
            affected = time_list[nan_mask]
            raise ValueError(
                f"在时间点 {list(affected)}，所有 nulls 均被拒绝，"
                f"置信区间边界全部为 NaN。"
                f"当前 nulls 范围 [{self.nulls[0]:.4f}, {self.nulls[-1]:.4f}] 太窄，"
                f"请扩大 nulls 范围（如 np.linspace(-50, 50, 100)）后重试。"
            )

        '3. 结果转化，将矩阵转化成dataframe'
        self.conformal_interval = self.result_to_dataframe(confidence_interval, time_list)

        return self.conformal_interval
    

    def fit(self, time_list, p_value_matrix):
        """
        完全共形推断计算流程
        """

        '第一层循环：时间'
        for i, treat_time_conformal in enumerate(tqdm(time_list, desc='p_value_matrix计算: ')):
            
            # 仅保留处理前数据 + 当前测试时间点。
            # 使用 `(< self.treat_time) | (== treat_time_conformal)` 而非 `<= treat_time_conformal`，
            # 确保增强数据集中只有 1 个处理后时期（当前测试期），避免多个处理后时期
            # 的 effect 互相污染置换分布。DID 在子样本中会将测试期识别为"处理发生年"，
            # 但这不影响共形推断——事件研究以 t=-1 为基期，测试的是该时间点的总缺口。
            conformal_data_pro_0 = self.data[(self.data[self.time] < self.treat_time) | (self.data[self.time] == treat_time_conformal)]

            # 提前计算目标行的布尔 mask（对所有 null 都相同，避免内层循环重复扫描）
            target_mask = conformal_data_pro_0[self.id].isin(self.target_id_list) & (conformal_data_pro_0[self.time] == treat_time_conformal)

            '第二层循环，遍历nulls对应行的每一个null'
            # 提前生成定长数组，提高运行速度
            p_value_list = np.zeros(len(self.nulls))
            for j, null in enumerate(self.nulls):
                # 采用copy的方式，让conformal_data_pro每次循环重置，不然会一直累加null
                conformal_data_pro = conformal_data_pro_0.copy()

                # 将增强数据集中目标行的 y 值减去 null
                conformal_data_pro.loc[target_mask, self.y_col] -= null
                
                # 使用增强数据训练模型（透传用户 kwargs 确保内外拟合规格一致）
                fit_econmodel_data = self.econ_model.fit_econmodel(data=conformal_data_pro, time=self.time, id=self.id, y_col=self.y_col, treat_col=self.treat_col, coverage=self.coverage, controls_col=self.controls_col, **self._econ_kwargs)

                # 校验返回结果格式
                self._validate_econ_results(fit_econmodel_data, context=f'Full(t={treat_time_conformal}, null={null:.4f})')

                # 获取处理效应
                residuals = fit_econmodel_data[['effect']]

                # 根据残差计算P值, 并存储在向量对应位置
                p_value_list[j] = self._get_pvalue_each_permutations(residuals)

            # 单周期循环完成，按行存储结果
            p_value_matrix[i] = p_value_list

        return p_value_matrix
    
    def predict(self):
        """为每个新样本计算预测区间"""
        if not hasattr(self, 'p_value_matrix'):
            raise RuntimeError(
                "尚未计算 p_value_matrix，请先调用 fit() 再调用 predict()。"
            )
        # 生成nulls矩阵,（行为p_value_matrix行数，列为nulls长度，每行都是nulls，方便找区间）
        nulls_matrix = np.tile(self.nulls, (self.p_value_matrix.shape[0], 1)).astype(float)

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
        time_list = np.array(sorted(self.data.loc[self.data[self.time] >= self.treat_time, self.time].unique()))

        # p值空矩阵，用于存储每个null的计算结果，提前生成矩阵提高运算速度。行数为处理后时期长度，列为nulls的长度       
        p_value_matrix = np.zeros((len(time_list), len(self.nulls)))

        return time_list, p_value_matrix

    def result_to_dataframe(self, confidence_interval, time_list):
        """将结果转换为DataFrame"""
        return pd.DataFrame(
                        confidence_interval,
                        index=time_list,
                        columns=[self.ci_lower_col, self.ci_upper_col])

    def _get_pvalue_each_permutations(self, resid_df):
        resid_df['post_intervention'] = False  # 先将所有值设为False
        # 将'post_intervention'列最后一个元素设置为True
        resid_df.iloc[-1, resid_df.columns.get_loc('post_intervention')] = True

        u = resid_df["effect"].values
        post_intervention = resid_df["post_intervention"].values

        # 注意：使用 np.roll 进行圆形置换（circular permutation）。
        # 该方法对无趋势的平稳时间序列近似有效，但对于存在单调趋势的面板数据，
        # 圆形移位会引入原始数据中不存在的人工模式（如 roll([1,2,3,4,5], 1)
        # = [5,1,2,3,4]），违反交换性假设。未来的改进可考虑使用块置换
        # （block permutation）来更好地保持时序依赖结构。
        block_permutations = np.stack([np.roll(u, permutation, axis=0)[post_intervention]
                                    for permutation in range(len(u))])
        
        statistics = self._test_statistic(block_permutations, q=1, axis=1)
        
        p_val = np.mean(statistics >= statistics[0])

        return p_val

    def _test_statistic(self, u_hat, q=1, axis=0):
        return (np.abs(u_hat) ** q).mean(axis=axis) ** (1/q)
    
