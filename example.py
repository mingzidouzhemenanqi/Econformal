
import pandas as pd
import numpy as np
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data


####### SC ########

# 用户初始化模型
data = pd.read_csv('ignore/data/smoking_panel.csv')
'''data = generate_test_panel_data(
        n_ids=200,
        n_treated=1,
        start_year=2010,
        pre_periods=15,
        post_periods=10,
        x_num=3,
        seed=123
    )'''
model = Econformal(data=data, time='year', id='state', y_col='cigsale', controls_col=[], treat_col='Treat')
# model = Econformal(data=data, time='year', id='id', y_col='Y', controls_col=[], treat_col='Treat')
nulls = np.linspace(-60, 20, 100)

# 计量模型拟合
result = model.conformal_inference(econ_model = 'sc', conformal_model='full', nulls=nulls, coverage=0.9)

print(result)

# 用户查看结果
fig = model.plot_ci_inteveral()
fig.show()



###### DID ######
"""data = generate_test_panel_data(
        n_ids=200,
        n_treated=1,
        start_year=2010,
        pre_periods=15,
        post_periods=10,
        x_num=3,
        seed=123
    )
# 读取ignore文件夹中data文件夹的smoking.csv文件

print(data.head())

# 用户初始化模型
# model = Econformal(data=data, time='year', id='id', y_col='Y', controls_col=['X1', 'X2', 'X3'], treat_col='Treat')
nulls = np.linspace(-10, 10, 100)

# 计量模型拟合
result = model.conformal_inference(econ_model = 'sc', conformal_model='full', nulls=nulls, coverage=0.90)

print(result)

# 用户查看结果
fig = model.plot_ci_inteveral()
fig.show()"""

"""SDID"""
"""
data = generate_test_panel_data(
        n_ids=200,
        n_treated=100,
        start_year=2010,
        pre_periods=15,
        post_periods=6,
        x_num=3,
        seed=123
    )
print(data.head())

# 用户初始化模型
model = Econformal(data=data, time='year', id='id', y_col='Y', controls_col=['X1', 'X2', 'X3'], treat_col='Treat')
nulls = np.linspace(-10, 10, 100)

# 计量模型拟合
result = model.conformal_inference(econ_model = 'SDID', conformal_model='full', nulls=nulls, coverage=0.90)
print(result)

# 用户查看结果
fig = model.plot_ci_inteveral(traditional=True)
fig.show()
"""