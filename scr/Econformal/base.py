
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
from matplotlib import style
style.use("ggplot")
from typing import Dict, Any

from .tools.model_registration import get_conformal_model, get_econ_model
from .tools import check, plot



class Econformal:
    def __init__(self, data: pd.DataFrame, time: str, id: str, y_col: str, x_cols: list, treat_col: str, **kwargs):
        """
        Econformal初始化，只要求输入数据，并明确变量含义
        data：数据表，要求dataframe
        time：时间变量名
        id：个体变量名
        y_col：因变量名
        x_cols：控制变量名
        treat_col：是否处理（0-1变量）
        """
        '''
        本函数仅初始化数据，并对数据格式进行检查，不进行任何拟合
        '''

        # 时间、个体、因变量、自变量、处理标识保存为类变量
        self.time = time  
        self.id = id   
        self.y_col = y_col
        self.x_cols = x_cols
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
        self.ci_upper_col = f"{int(coverage*100)}%_ci_upper"
        self.ci_lower_col = f"{int(coverage*100)}%_ci_lower"

        ########### 执行计量经济学模型拟合 ##########
        self.econ_results = self.econ_fit(econ_model=econ_model, coverage = coverage)    # 该方法最后生成类变量self.econ_results

        ########### 执行共形推断进行区间预测 ##########
        self.conformal_interval = self.conformal_inference_fit(econ_model=econ_model, conformal_model=conformal_model, nulls=nulls, coverage=coverage)    # 该方法最后生成类变量self.conformal_interval

        ########### 将共形推断预测区间与econ_results合并 ##########
        self.results = self._merge_results()

        return self.results

    """计量经济学模型拟合方法"""
    def econ_fit(self, econ_model: str, coverage: float, **kwargs):
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
        std_error    暂时为0
        p-value      暂时为0
        置信区间下界  暂时为0
        置信区间上界  暂时为0
        """
        # 拟合模型模式：0:仅使用训练集训练数据；1:使用所有样本数据训练。默认为1
        train_mode = 1
        if econ_model == 'SC':
            train_mode = 0   # SC模型仅需要处理前数据进行训练，因此需要将train_mode设为0

        econ_results = self.econ_model.fit_econmodel(
                            data=self.data,
                            time=self.time,
                            id=self.id,
                            y_col=self.y_col,
                            treat_col=self.treat_col,
                            x_cols=self.x_cols,
                            coverage=coverage,
                            train_mode = train_mode # 0:仅使用训练集训练数据；1:使用所有样本数据训练。默认为1
                        )
        

        return econ_results
        
    def conformal_inference_fit(self, conformal_model: str, coverage: float, nulls:list=None, **kwargs):
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

        # 将index的值作为time列
        self.conformal_interval[self.time] = self.conformal_interval.index
        # 重置index
        self.conformal_interval.reset_index(drop=True, inplace=True)

        return pd.merge(self.conformal_interval, self.econ_results, on=self.time, how='right')
