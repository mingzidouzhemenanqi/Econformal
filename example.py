
import pandas as pd
import numpy as np
import scr.Econformal

from scr.Econformal.tools.generate_data import generate_test_panel_data


####### SC ########
'''
# 用户初始化模型
model = scr.Econformal.base.Econformal(data=data, time='year', id='state', y_col='cigsale', x_cols=[], treat_col='treat')
nulls = np.linspace(-60, 20, 100)

# 计量模型拟合
result = model.conformal_inference(econ_model = 'SC', conformal_model='full', nulls=nulls, coverage=0.9)

print(result)

# 用户查看结果
model.plot_ci_inteveral()'''

###### DID ######
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
model = scr.Econformal.base.Econformal(data=data, time='year', id='id', y_col='Y', x_cols=['X1', 'X2', 'X3'], treat_col='Treat')
nulls = np.linspace(-10, 10, 100)

# 计量模型拟合
result = model.conformal_inference(econ_model = 'DID', conformal_model='Full', nulls=nulls, coverage=0.90)

print(result)

# 用户查看结果
fig = model.plot_ci_inteveral()
fig.show()

"""SDID"""
"""
data = scr.Econformal.tools.generate_data.generate_test_panel_data(
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
model = scr.Econformal.base.Econformal(data=data, time='year', id='id', y_col='Y', x_cols=['X1', 'X2', 'X3'], treat_col='Treat')
nulls = np.linspace(-10, 10, 100)

# 计量模型拟合
result = model.conformal_inference(econ_model = 'SDID', conformal_model='Full', nulls=nulls, coverage=0.90)
print(result)

# 用户查看结果
fig = model.plot_ci_inteveral(traditional=True)
fig.show()
"""