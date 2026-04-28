import pandas as pd
import numpy as np
import econformal

from econformal.tools import generate_data


#n_zises = [4,100,200]
#pre_periods = [5,15,30]
#coverages = [0.9,0.95,0.99]

n_zises = [i for i in range(10, 101, 10)]
pre_periods = [20]
coverages = [0.9]

for coverage_value in coverages:
    for n_zise in n_zises:
        for pre_period in pre_periods:
            data = econformal.tools.generate_data.generate_test_panel_data(
                n_ids=n_zise,
                n_treated=int(n_zise*0.5),
                start_year=2010,
                pre_periods=pre_period,
                post_periods=8,
                x_num=5,
                seed=123
            )

            # 用户初始化模型
            model = econformal.base.Econformal(data=data, time='year', id='id', y_col='Y', x_cols=['X1', 'X2', 'X3'], treat_col='Treat')
            nulls = np.linspace(-5, 5, 100)

            # 计量模型拟合
            result = model.conformal_inference(econ_model = 'DID', conformal_model='Full', nulls=nulls, coverage=coverage_value)
            
            # 用户查看结果
            fig = model.plot_ci_inteveral(traditional=True)
            infor = f'总样本数={n_zise}, 处理组样本数={int(n_zise*0.5)}, 处理前期数={pre_period}, 置信水平={coverage_value}'
            fig.figtext(0.5, 0.01, infor, fontsize=9, va='bottom')
            fig.savefig(rf'C:\Users\Forry\Desktop\plt\总样本数={n_zise}, 处理组样本数={int(n_zise*0.5)}, 处理前期数={pre_period}, 置信水平={coverage_value}.jpg')