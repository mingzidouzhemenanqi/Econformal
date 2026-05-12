import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
from matplotlib import style
style.use("ggplot")

from .tools.model_registration import get_conformal_model, get_econ_model
from .tools import check, plot



class Econformal:
    def __init__(self, data: pd.DataFrame, time: str, id: str, y_col: str, controls_col: list, treat_col: str, **kwargs):
        """
        Econformal初始化，只要求输入数据，并明确变量含义
        data：数据表，要求dataframe
        time：时间变量名
        id：个体变量名
        y_col：因变量名
        controls_col：控制变量名
        treat_col：是否处理（0-1变量）
        """
        '''
        本函数仅初始化数据，并对数据格式进行检查，不进行任何拟合
        '''

        # 时间、个体、因变量、自变量、处理标识保存为类变量
        self.time = time  
        self.id = id   
        self.y_col = y_col
        self.controls_col = controls_col
        self.treat_col = treat_col

        # 检查日期是否为int或者时间等可排序类型

        # 检查数据格式，并重新排序
        data = self._validate_data_format(data)

        # 识别处理个体,并生成样本识别代码
        data = check.get_id_code(data, id)

        # 检查treat_col列是否只包含0-1，同时检查0-1是否都有


        # 预处理完成的数据列表保存为类变量
        self.data = data


    def conformal_inference(self, econ_model: str, conformal_model: str, nulls:list, coverage: float = 0.9, **kwargs):

        """
        与用户交互的主要方法
        用户调用该方法所需输入：
            必选参数：计量模型类型，共形推断模型类型，原假设列表（暂时必选，后续增加根据数据表自动生成coverage列表的方法）
            可选参数：覆盖率，

        返回结果格式：dataframe
                year
                prediction   如果是DID这种直接出effcet的方法，为0
                effect
                std_error    暂时为0
                p-value      暂时为0
                置信区间下界  暂时为0
                置信区间上界  暂时为0
                共形推断区间下界
                共形推断区间上界
        """
        ############ 检查 ##########
        # 用户输入的计量模型是否可以使用

        # 用户输入的共形推断模型是否可以使用


        ########### 置信区间范围参数预设 ##########
        # 覆盖率保存为类变量
        self.coverage = coverage
        # 预先设置置信区间列名

        # 预先设置共形推断区间列名
        self.ci_lower_col = f"{int(self.coverage * 100)}%_conformal_lower"
        self.ci_upper_col = f"{int(self.coverage * 100)}%_conformal_upper"

        ########### 执行计量经济学模型拟合 ##########
        self.econ_results = self.econ_fit(econ_model=econ_model, coverage = self.coverage)    # 该方法最后生成类变量self.econ_results

        ########### 执行共形推断进行区间预测 ##########
        self.conformal_interval = self.conformal_inference_fit(econ_model=econ_model, conformal_model=conformal_model, nulls=nulls)    # 该方法最后生成类变量self.conformal_interval

        ########### 将共形推断预测区间与econ_results合并 ##########
        self.results = self._merge_results()

        return self.results

    """计量经济学模型拟合方法"""
    def econ_fit(self, econ_model: str, **kwargs):
        '''
        econ_model: 需要调用的计量经济模型
        y_col: 因变量列名

        treat_time: 处理时间，适用于SC
        target_id: 目标个体ID，适用于SC

        econ_fit流程：
        1. 根据用户输入的模型，调用相应的计量经济学模型
        2. 返回模型结果，存于self.predictions
        '''

        ############ econ模型注册 ##########
        ModelClass = get_econ_model(econ_model)
        self.econ_model = ModelClass()   # econ模型初始化

        ############ econ模型拟合，并返回预测结果 ##########
        """
        返回数据格式dataframe
        year
        event_time   处理前1期为-1
        prediction   如果是DID这种直接出effcet的方法，为0
        effect
        std_error
        p-value
        置信区间下界
        置信区间上界
        """
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
        
    def conformal_inference_fit(self, conformal_model: str, nulls:list=None, **kwargs):
        """
        执行共形推断计算置信区间
        0. 获取误差率，并检查是否已经拟合模型
        1. 检查nulls参数是否为None，若为None，则调用get_nulls()方法获取nulls
        2. 根据用户输入的模型，调用相应的共形推断模型
        3. 返回模型结果，存于self.predictions
        """
        #if self.predictions is None:
        #    raise RuntimeError("请先使用econ_fit()方法拟合计量模型")
        
        'nulls参数'
        # 暂时没想好nulls如何自动生成，先保留位置，并将nulls定为必填参数
        
        '根据用户输入的模型，注册相应的共形推断模型，并计算预测区间'
        # 模型注册
        ConformalClass = get_conformal_model(conformal_model)
        # 共形推断模型初始化
        self.conformal_model = ConformalClass(econ_model=self.econ_model, 
                                            data=self.data,
                                            time=self.time,
                                            id=self.id,
                                            y_col=self.y_col,
                                            nulls=nulls,
                                            treat_col=self.treat_col,
                                            coverage=self.coverage, 
                                            )
        # 共形推断计算
        conformal_interval = self.conformal_model.compute_conformal_interval()
        
        return conformal_interval
    
    def plot_ci_inteveral(self, traditional=False, y_label=None, **kwargs):
        """绘制结果及置信区间"""
        if self.conformal_interval is None:
            raise RuntimeError("请先使用conformal_inference()计算共形推断置信区间")

        if traditional:
            # 绘制共形预测区间和传统置信区间
            plt=plot.ci_interval_compare(self, **kwargs)
        else:
            # 调用函数绘制图像
            plt = plot.ci_interval(self)
        return plt

    # 数据格式检查
    def _validate_data_format(self, data):
        '''
        检查内容：
        1. 检查data是否为强面板数据
        2. 检查data列名是否包含'T_'
        '''
        # 检查数据是否为强面板数据
        # 若是，打印'The data set is a strongly balanced panel data'; 若不是，抛出warining
        check.strong_panel(data=data, id_col=self.id, time_col = self.time)

        #检查data列名是否包含'T_'
        #若包含，抛出错误，提示：列名不能包含'T_'，否则影响DID等模型的估计
        check.data_col_name_check_T_(data=data)

        # data根据time列和id列排序, 并reset_index
        data = data.sort_values(by=[self.time, self.id]).reset_index(drop=True)

        return data
    
    def _merge_results(self):
        """
        合并共形推断区间和计量模型结果
        
        注意：该方法会直接修改 self.conformal_interval（添加time列并重置索引）
        这是设计意图，因为合并后不再需要原始的 conformal_interval
        """
        
        # ========== 1. 空值检查 ==========
        if self.conformal_interval is None:
            raise RuntimeError("请先调用 conformal_inference_fit() 计算共形推断区间")
        
        if self.econ_results is None:
            raise RuntimeError("请先调用 econ_fit() 拟合计量模型")
        
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
        
        # ========== 3. 添加时间列（直接修改，节省内存）==========
        self.conformal_interval[self.time] = self.conformal_interval.index
        self.conformal_interval.reset_index(drop=True, inplace=True)
        
        # ========== 4. 验证 econ_results ==========
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
        
        # ========== 5. 统一时间列数据类型 ==========
        conformal_dtype = self.conformal_interval[self.time].dtype
        econ_dtype = self.econ_results[self.time].dtype
        
        if conformal_dtype != econ_dtype:
            # 尝试统一类型
            unified_dtype = self._infer_common_time_dtype(conformal_dtype, econ_dtype)
            
            import warnings
            warnings.warn(
                f"共形推断区间和计量模型结果的时间列类型不一致：\n"
                f"  - conformal_interval['{self.time}']: {conformal_dtype}\n"
                f"  - econ_results['{self.time}']: {econ_dtype}\n"
                f"将自动统一为: {unified_dtype}"
            )
            
            self.conformal_interval[self.time] = self.conformal_interval[self.time].astype(unified_dtype)
            self.econ_results[self.time] = self.econ_results[self.time].astype(unified_dtype)
        
        # ========== 6. 执行合并 ==========
        return pd.merge(
            self.conformal_interval,
            self.econ_results,
            on=self.time,
            how='outer'
        )

    def _infer_common_time_dtype(self, dtype1, dtype2):
        """
        推断两个时间列的公共数据类型
        
        规则：
        1. 如果都是数值类型，使用精度更高的
        2. 如果一个是数值一个是字符串，优先使用字符串（保留原始格式）
        3. 其他情况使用 object
        """
        
        # 都是数值类型
        if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
            return np.result_type(dtype1, dtype2)
        
        # 一个是数值，一个是字符串/对象
        if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_string_dtype(dtype2):
            return dtype2  # 使用字符串类型
        
        if pd.api.types.is_string_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
            return dtype1  # 使用字符串类型
        
        # 默认使用 object
        return object
