import pandas as pd

import warnings
warnings.filterwarnings('ignore')
from matplotlib import style
style.use("ggplot")

from .tools.model_registration import get_conformal_model, get_econ_model
from .tools import check, plot



class Econformal:
    def __init__(self, data: pd.DataFrame, time: str, id: str, y_col: str, treat_col: str, controls_col: list[str] | None = None):
        """初始化并校验数据，不进行任何拟合。

        Parameters
        ----------
        data : pd.DataFrame
            面板数据表
        time : str
            时间变量名
        id : str
            个体变量名
        y_col : str
            因变量名
        controls_col : list, optional
            控制变量名列表
        treat_col : str
            是否处理（0-1变量）
        """
        self.time = time
        self.id = id
        self.y_col = y_col
        self.controls_col = controls_col if controls_col is not None else []
        self.treat_col = treat_col

        check.validate_data_format(data, self.time, self.id, self.treat_col)
        self.data = check.preprocess_data(data, self.id, self.time)


    def conformal_inference(self, econ_model: str, conformal_model: str,
                            coverage: float = 0.9,
                            nulls: list = None, split_rate=0.7, **kwargs):

        """
        与用户交互的主要方法
        用户调用该方法所需输入：
            必选参数：计量模型类型，共形推断模型类型，原假设列表（暂时必选，后续增加根据数据表自动生成nulls列表的方法）
            可选参数：覆盖率，

        返回结果格式：dataframe
                year
                prediction   
                effect
                std_error    
                p-value      
                90%_conf_lower
                90%_conf_upper
                90%_conformal_lower
                90%_conformal_upper
        """
        # 校验用户输入的模型是否可用，并获取模型类
        EconModelCls = get_econ_model(econ_model)
        ConformalCls = get_conformal_model(conformal_model)

        ########### 置信区间范围参数预设 ##########
        check.validate_coverage(coverage)
        self.coverage = coverage

        # 预设共形推断区间列名
        self.ci_lower_col = f"{int(self.coverage * 100)}%_conformal_lower"
        self.ci_upper_col = f"{int(self.coverage * 100)}%_conformal_upper"

        ########### 执行计量经济学模型拟合 ##########
        self.econ_results = self._econ_fit(EconModelCls)

        ########### 执行共形推断进行区间预测 ##########
        self.conformal_interval = self._conformal_inference_fit(ConformalCls,
                                                                nulls=nulls,
                                                                split_rate=split_rate,
                                                                **kwargs)

        ########### 将共形推断预测区间与econ_results合并 ##########
        self.results = self._merge_results()

        return self.results

    def plot_ci_interval(self, traditional=False):
        """绘制结果及置信区间"""
        if getattr(self, 'results', None) is None:
            raise RuntimeError("请先使用conformal_inference()计算共形推断置信区间")

        if traditional:
            fig = plot.ci_interval_compare(self)
        else:
            fig = plot.ci_interval(self)
        return fig


    def _econ_fit(self, econ_model_cls):
        """拟合计量经济学模型，结果存于 self.econ_results。"""
        self.econ_model = econ_model_cls()

        # 拟合模型并返回预测结果
        econ_results = self.econ_model.fit_econmodel(
                            data=self.data,
                            time=self.time,
                            id=self.id,
                            y_col=self.y_col,
                            treat_col=self.treat_col,
                            controls_col=self.controls_col,
                            coverage=self.coverage,
                        )

        return econ_results
        
    def _conformal_inference_fit(self, conformal_model_cls, nulls=None, split_rate=None, **kwargs):
        """执行共形推断计算置信区间，结果存于 self.conformal_interval。"""

        # nulls参数：暂时没想好nulls如何自动生成，先将nulls定为必填参数
        self.conformal_model = conformal_model_cls(econ_model=self.econ_model,
                                            data=self.data,
                                            time=self.time,
                                            id=self.id,
                                            y_col=self.y_col,
                                            treat_col=self.treat_col,
                                            coverage=self.coverage,
                                            controls_col=self.controls_col,
                                            nulls=nulls,  # 原假设列表，仅full使用
                                            split_rate=split_rate,
                                            econ_results=self.econ_results,
                                            **kwargs
                                            )
        conformal_interval = self.conformal_model.compute_conformal_interval()

        return conformal_interval
    

    def _merge_results(self):
        """
        合并共形推断区间和计量模型结果
        
        注意：该方法会直接修改 self.conformal_interval（添加time列并重置索引）
        这是设计意图，因为合并后不再需要原始的 conformal_interval
        """
        
        # ========== 1. 空值检查 ==========
        if self.conformal_interval is None:
            raise RuntimeError("conformal_interval 为 None，请通过 conformal_inference() 调用")

        if self.econ_results is None:
            raise RuntimeError("econ_results 为 None，请通过 conformal_inference() 调用")
        
        # ========== 2. 验证 conformal_interval ==========
        if self.conformal_interval.empty:
            raise ValueError("共形推断区间为空，无法合并")

        if self.conformal_interval.index.has_duplicates:
            raise ValueError(
                f"共形推断区间的索引（时间列）存在重复值，这会导致合并结果异常。"
                f"重复值: {self.conformal_interval.index[self.conformal_interval.index.duplicated()].unique()}"
            )

        if self.conformal_interval.index.isna().any():
            raise ValueError("共形推断区间的索引（时间列）包含 NaN 值")

        # ========== 3. 验证 econ_results ==========
        if self.econ_results.empty:
            raise ValueError("计量模型结果为空，无法合并")

        if self.time not in self.econ_results.columns:
            raise ValueError(
                f"计量模型结果中缺少时间列 '{self.time}'。"
                f"当前列名: {list(self.econ_results.columns)}"
            )

        if self.econ_results[self.time].duplicated().any():
            raise ValueError(
                f"计量模型结果的时间列 '{self.time}' 存在重复值，这会导致合并结果异常。"
                f"重复值: {self.econ_results[self.time][self.econ_results[self.time].duplicated()].unique()}"
            )

        if self.econ_results[self.time].isna().any():
            raise ValueError(f"计量模型结果的时间列 '{self.time}' 包含 NaN 值")

        # ========== 4. 添加时间列（直接修改，节省内存）==========
        self.conformal_interval[self.time] = self.conformal_interval.index
        self.conformal_interval.reset_index(drop=True, inplace=True)
        
        # ========== 5. 执行合并 ==========
        return pd.merge(
            self.conformal_interval,
            self.econ_results,
            on=self.time,
            how='outer'
        )
