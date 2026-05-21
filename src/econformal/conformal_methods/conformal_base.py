from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from ..tools import check


class ConformalBase(ABC):
    """共形推断算法的抽象基类。

    所有共形推断实现（Full、Split、Jackknife+、CV+）共享相同的数据接口
    和主入口方法 compute_conformal_interval()。子类只需实现具体的拟合和预测逻辑。

    子类必须实现：
        - compute_conformal_interval(): 主入口方法，调用预处理、拟合、预测、格式化结果
        - fit(): 计算共形推断所需的核心统计量（分位数/p值矩阵/残差分布等）
        - predict(): 基于fit()的结果生成置信区间

    可选覆盖：
        - preprocess_data(): 在fit()前准备数据，默认提取处理信息
        - result_to_dataframe(): 将预测结果转为DataFrame，默认格式与合并逻辑兼容
    """


    def __init__(self, econ_model, data: pd.DataFrame, time: str, id: str,
                y_col: str, treat_col: str, coverage: float, 
                controls_col: list=[], nulls: list=[], **kwargs):
        self.econ_model = econ_model   # 计量模型
        self.coverage = coverage     # 置信区间覆盖率
        self.data = data            # 输入的数据集
        self.time = time              # 时间列名
        self.id = id                  # 个体标识列名
        self.y_col = y_col              # 因变量列名
        self.treat_col = treat_col      # 是否处理列名
        self.controls_col = controls_col              # 控制变量列名列表

        # 置信区间列名（与 _merge_results 的列名约定保持一致）
        self.ci_lower_col = f"{int(self.coverage * 100)}%_conformal_lower"
        self.ci_upper_col = f"{int(self.coverage * 100)}%_conformal_upper"

        # 从数据中识别处理组个体和首次处理时间
        self.target_id_list = check.get_treated_individuals(
            data=data, id=id, time=time, treat_col=treat_col)
        self.treat_time = check.get_first_treatment_year(
            data=data, id=id, time=time, treat_col=treat_col)


    @abstractmethod
    def compute_conformal_interval(self, **kwargs):
        """共形推断主流程：预处理 → 拟合 → 预测 → 格式化结果。

        这是 Econformal.conformal_inference_fit() 调用的唯一入口。
        """

        pass

    @abstractmethod
    def preprocess_data(self, **kwargs):
        """准备 fit() 所需的数据。"""

        pass

    @abstractmethod
    def fit(self, **kwargs):
        """计算共形推断的核心统计量。

        子类应将计算结果存储为实例属性（如 self.quantile、self.p_value_matrix 等），
        供 predict() 使用。返回 self 以支持链式调用。
        """
        pass

    @abstractmethod
    def predict(self, **kwargs):
        """基于 fit() 的结果生成预测区间。

        返回格式不限制（数组、DataFrame等），最终由 result_to_dataframe() 统一转换。
        """
        pass

    @abstractmethod
    def result_to_dataframe(self, **kwargs):
        """将 predict() 的输出转为标准 DataFrame 格式。

        """
        pass 
