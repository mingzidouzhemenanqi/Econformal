import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False 

import warnings
warnings.filterwarnings('ignore')

from matplotlib import style
style.use("ggplot")
from . import check


def ci_interval(self_data):
    
    # 读取结果数据表
    plot_data = self_data.results
    # 读取首次处理时间
    treat_time = check.get_first_treatment_year(data=self_data.data, id=self_data.id, time=self_data.time, treat_col=self_data.treat_col)
    # 将self_data.time列设置为index
    plot_data.set_index(self_data.time, inplace=True)

    # 图像时间范围
    # 起始时间：pre_data的index最小值
    start_time = plot_data.index.min()
    # 结束时间：pre_data的index最大值
    end_time = plot_data.index.max()


    # 政策效应范围
    # pre_data中residuals列最大值和ci_df中f"{int(plot_data.coverage*100)}%_ci_upper"列最大值中取最大值
    max_y = max(plot_data.effect.max(), plot_data[self_data.ci_upper_col].max())
    # pre_data中residuals列最小值和ci_df中f"{int(plot_data.coverage*100)}%_ci_lower"列最小值中取最小值
    min_y = min(plot_data.effect.min(), plot_data[self_data.ci_lower_col].min())


    
    # 图片画幅
    plt.figure(figsize=(10,5))

    # 绘制政策效应共性预测区间
    plt.fill_between(plot_data.index, plot_data[self_data.ci_lower_col], plot_data[self_data.ci_upper_col], alpha=0.2,  color="C1")
    # 绘制残差曲线
    plt.plot(plot_data.index, plot_data["effect"], label='Effect', color="C1")

    # 绘制横轴
    plt.hlines(y=0, xmin=start_time, xmax=end_time, lw=2, color="Black")
    # 绘制纵轴
    plt.vlines(x=treat_time, ymin=min_y, ymax=max_y, linestyle=":", color="Black", lw=2, label='Treat')
    # 图例
    plt.legend()
    # 在图例中添加alpha=self_data.coverage%
    plt.legend(title=f"Coverage: {int(self_data.coverage*100)}%")
    # y轴标题
    # plt.ylabel("")
    
    # 将图片保存到工作路径
    # plt.savefig(f"{plot_data.y_col},alpha={int(plot_data.coverage*100)}%.png")
    # plt.show()

    return plt


def ci_interval_compare(self_data):
    
    # 读取结果数据表
    plot_data = self_data.results
    # 读取首次处理时间
    treat_time = check.get_first_treatment_year(data=self_data.data, id=self_data.id, time=self_data.time, treat_col=self_data.treat_col)
    # 将self_data.time列设置为index
    plot_data.set_index(self_data.time, inplace=True)

    # 图像时间范围
    # 起始时间：pre_data的index最小值
    start_time = plot_data.index.min()
    # 结束时间：pre_data的index最大值
    end_time = plot_data.index.max()


    # 政策效应范围
    # pre_data中residuals列最大值和ci_df中f"{int(plot_data.coverage*100)}%_ci_upper"列最大值中取最大值
    max_y = max(plot_data.effect.max(), plot_data[self_data.ci_upper_col].max())
    # pre_data中residuals列最小值和ci_df中f"{int(plot_data.coverage*100)}%_ci_lower"列最小值中取最小值
    min_y = min(plot_data.effect.min(), plot_data[self_data.ci_lower_col].min())


    
    # 图片画幅
    plt.figure(figsize=(10,5))

    # 绘制政策效应共性预测区间
    plt.fill_between(plot_data.index, plot_data[self_data.ci_lower_col], plot_data[self_data.ci_upper_col], alpha=0.2,  color="C1")

    # 绘制残差曲线
    plt.plot(plot_data.index, plot_data["effect"], label='Effect', color="C1")

    # 绘制置信区间线段（使用误差棒）
    time_points = plot_data.index.values
    for t in time_points:
        row = plot_data.loc[t]
        plt.plot([t, t], [row['置信区间下界'], row['置信区间上界']], 
                color='gray', linewidth=1.5, alpha=0.8)
        # 在置信区间两端添加小横线
        plt.plot([t-0.1, t+0.1], [row['置信区间下界'], row['置信区间下界']], 
                color='gray', linewidth=1.5, alpha=0.8)
        plt.plot([t-0.1, t+0.1], [row['置信区间上界'], row['置信区间上界']], 
                color='gray', linewidth=1.5, alpha=0.8)
    # 绘制横轴
    plt.hlines(y=0, xmin=start_time, xmax=end_time, lw=2, color="Black")
    # 绘制纵轴
    plt.vlines(x=treat_time, ymin=min_y, ymax=max_y, linestyle=":", color="Black", lw=2, label='Treat')
    # 图例
    plt.legend()
    # 在图例中添加alpha=self_data.coverage%
    plt.legend(title=f"Coverage: {int(self_data.coverage*100)}%")
    # y轴标题
    # plt.ylabel("")
    
    # 将图片保存到工作路径
    # plt.savefig(rf'C:\Users\Forry\Desktop\plt\{},alpha={int(plot_data.coverage*100)}%.png"')
    # plt.show()

    # return 0
    return plt