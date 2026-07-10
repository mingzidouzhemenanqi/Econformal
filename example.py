import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data


# 查看可用模型
Econformal.get_models()


###### SC + Full Conformal ######
print("\n" + "=" * 50)
print("SC + Full Conformal")
print("=" * 50)
data = generate_test_panel_data(n_ids=30, n_treated=1, pre_periods=10, post_periods=5, x_num=0, seed=99)
model = Econformal(data=data, time='year', id='id', y_col='Y', treat_col='Treat')
nulls = np.linspace(-10, 10, 50)
result = model.conformal_inference(econ_model='sc', conformal_model='full', nulls=nulls, coverage=0.9)
print(result.round(4).to_string())
# model.plot_ci_interval()
# plt.show()


###### DID + Split Conformal ######
print("\n" + "=" * 50)
print("DID + Split Conformal")
print("=" * 50)
data = generate_test_panel_data(n_ids=200, n_treated=100, pre_periods=15, post_periods=10, x_num=3, seed=123)
model = Econformal(data=data, time='year', id='id', y_col='Y',
                   controls_col=['X1', 'X2', 'X3'], treat_col='Treat')
result = model.conformal_inference(econ_model='did', conformal_model='split', coverage=0.9)
print(result.round(4).to_string())
# model.plot_ci_interval()
# plt.show()


###### SDID + CV+ Conformal ######
print("\n" + "=" * 50)
print("SDID + CV+ Conformal")
print("=" * 50)
data = generate_test_panel_data(n_ids=200, n_treated=100, pre_periods=15, post_periods=6, x_num=3, seed=123)
model = Econformal(data=data, time='year', id='id', y_col='Y',
                   controls_col=['X1', 'X2', 'X3'], treat_col='Treat')
result = model.conformal_inference(econ_model='sdid', conformal_model='cv+', cv_folds=5, coverage=0.9)
print(result.round(4).to_string())
# model.plot_ci_interval()
# plt.show()


###### DID + LOO Conformal ######
print("\n" + "=" * 50)
print("DID + LOO Conformal")
print("=" * 50)
data = generate_test_panel_data(n_ids=200, n_treated=100, pre_periods=15, post_periods=10, x_num=3, seed=123)
model = Econformal(data=data, time='year', id='id', y_col='Y',
                   controls_col=['X1', 'X2', 'X3'], treat_col='Treat')
result = model.conformal_inference(econ_model='did', conformal_model='loo', coverage=0.9)
print(result.round(4).to_string())
# model.plot_ci_interval()
# plt.show()


###### SDID + JK+ Conformal ######
print("\n" + "=" * 50)
print("SDID + JK+ Conformal")
print("=" * 50)
data = generate_test_panel_data(n_ids=200, n_treated=100, pre_periods=15, post_periods=6, x_num=3, seed=123)
model = Econformal(data=data, time='year', id='id', y_col='Y',
                   controls_col=['X1', 'X2', 'X3'], treat_col='Treat')
result = model.conformal_inference(econ_model='sdid', conformal_model='jk+', coverage=0.9)
print(result.round(4).to_string())
# model.plot_ci_interval()
# plt.show()
