
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data


###### SC + Full Conformal ######
print("=" * 50)
print("SC + Full Conformal")
print("=" * 50)
data_sc = generate_test_panel_data(
    n_ids=30, n_treated=1, start_year=2000,
    pre_periods=10, post_periods=5, x_num=0, seed=99
)
model = Econformal(data=data_sc, time='year', id='id', y_col='Y', treat_col='Treat')
nulls = np.linspace(-10, 10, 50)
result = model.conformal_inference(econ_model='sc', conformal_model='full', nulls=nulls, coverage=0.9)
print(result)
fig = model.plot_ci_interval()
plt.show()


###### DID + Full Conformal ######
print("=" * 50)
print("DID + Full Conformal")
print("=" * 50)
data_did = generate_test_panel_data(
    n_ids=200, n_treated=1, start_year=2010,
    pre_periods=15, post_periods=10, x_num=3, seed=123
)
model = Econformal(data=data_did, time='year', id='id', y_col='Y',
                   controls_col=['X1', 'X2', 'X3'], treat_col='Treat')
nulls = np.linspace(-10, 10, 100)
result = model.conformal_inference(econ_model='did', conformal_model='full', nulls=nulls, coverage=0.90)
print(result)
fig = model.plot_ci_interval(traditional=True)
plt.show()


###### SDID + Full Conformal ######
print("=" * 50)
print("SDID + Full Conformal")
print("=" * 50)
data_sdid = generate_test_panel_data(
    n_ids=200, n_treated=100, start_year=2010,
    pre_periods=15, post_periods=6, x_num=3, seed=123
)
model = Econformal(data=data_sdid, time='year', id='id', y_col='Y',
                   controls_col=['X1', 'X2', 'X3'], treat_col='Treat')
nulls = np.linspace(-10, 10, 100)
result = model.conformal_inference(econ_model='sdid', conformal_model='full', nulls=nulls, coverage=0.90)
print(result)
fig = model.plot_ci_interval(traditional=True)
plt.show()
