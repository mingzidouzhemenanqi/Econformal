import numpy as np
import pandas as pd

def generate_test_panel_data(n_ids=100, 
                        n_treated=None,
                        start_year=2000,
                        pre_periods=3,
                        post_periods=2,
                        x_num=2,
                        seed=None,
                        base_effect = 5.0,  # 基准效应
                        treatment_effect = 2.0  # 处理效应大小
                        ):
    """
    生成面板格式的模拟数据
    
    参数:
    n_ids (int): 总个体数量 (默认100)
    n_treated (int): 处理组个体数量 (默认总个体的50%)
    start_year (int): 起始年份 (默认2000)
    pre_periods (int): 处理前期数 (默认3)
    post_periods (int): 处理后期数 (默认2)
    x_num (int): 自变量数量 (默认2)
    seed (int): 随机种子 (默认None)
    
    返回:
    pandas.DataFrame: 包含以下列的面板数据:
        year: 年份
        id: 个体ID
        Y: 因变量
        Treat: 处理状态（0=控制组，1=处理组）
        Group: 分组标识（"Control"或"Treatment"）
        X1, X2, ..., Xn: 自变量 (n=x_num)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # 设置处理组个体数（默认为总个体的一半）
    if n_treated is None:
        n_treated = max(1, n_ids // 2)  # 确保至少有一个处理组
    elif n_treated > n_ids:
        raise ValueError("处理组个体数不能超过总个体数")
    
    # 1. 基本参数计算
    total_years = pre_periods + post_periods
    years = np.arange(start_year, start_year + total_years)
    ids = np.arange(1, n_ids + 1)
    
    # 2. 创建基本面板结构
    df = pd.DataFrame(
        [(year, id_) for id_ in ids for year in years],
        columns=['year', 'id']
    )
    
    # 3. 确定处理组和控制组
    # 随机选择处理组个体
    treated_ids = np.random.choice(ids, size=n_treated, replace=False)
    df['Group'] = df['id'].apply(lambda x: 'Treatment' if x in treated_ids else 'Control')
    
    # 4. 生成处理变量
    treatment_year = start_year + pre_periods
    df['Treat'] = ((df['year'] >= treatment_year) & (df['Group'] == 'Treatment')).astype(int)
    
    # 5. 生成自变量 (X1, X2, ..., X_num)
    # 固定效应映射
    indiv_map = {}
    for id_ in ids:
        if id_ in treated_ids:
            # 处理组个体效应更强
            indiv_map[id_] = np.random.normal(1, 1.5)
        else:
            # 控制组个体效应
            indiv_map[id_] = np.random.normal(0, 1)
    
    # 应用个体固定效应
    for i in range(1, x_num + 1):
        # 复制基础个体效应并添加扰动
        df[f'X{i}'] = df['id'].map(indiv_map) + np.random.normal(0, 0.5, size=len(df))
        
        # 添加时间趋势 - 按组应用不同趋势
        time_trend = np.zeros(len(df))
        for idx, row in df.iterrows():
            year_idx = row['year'] - start_year
            if row['Group'] == 'Control':
                time_trend[idx] = 0.8 * (year_idx / (total_years - 1))
            else:
                time_trend[idx] = 1.2 * (year_idx / (total_years - 1))
        
        df[f'X{i}'] += time_trend
    
    # 6. 生成因变量 Y

    
    # 基础公式: Y = 基准效应 + 处理效应 + ∑(自变量系数 * 自变量)
    df['Y'] = base_effect + treatment_effect * df['Treat']
    
    # 添加自变量影响
    x_coeffs = np.random.uniform(-1, 1, size=x_num)
    for i, coef in enumerate(x_coeffs, 1):
        df['Y'] += coef * df[f'X{i}']
    
    # 添加个体固定效应 - 处理组和控制组有不同的分布
    indiv_y_map = {}
    for id_ in ids:
        if id_ in treated_ids:
            indiv_y_map[id_] = np.random.normal(0.5, 1.5)
        else:
            indiv_y_map[id_] = np.random.normal(0, 1)
    
    df['Y'] += df['id'].map(indiv_y_map)
    
    # 添加时间趋势 - 处理组和控制组有不同的趋势
    time_trend_y = np.zeros(len(df))
    for idx, row in df.iterrows():
        year_idx = row['year'] - start_year
        if row['Group'] == 'Control':
            time_trend_y[idx] = 0.3 * (year_idx / (total_years - 1))
        else:
            time_trend_y[idx] = 0.6 * (year_idx / (total_years - 1))
    
    df['Y'] += time_trend_y
    
    # 添加随机噪声
    df['Y'] += np.random.normal(0, 1, size=len(df))
    # 删除Group列
    df = df.drop(columns=['Group'])
    
    return df

if __name__ == "__main__":

    # 创建模拟数据（使用之前生成面板数据的函数）
    print("\n==== 步骤0: 模拟数据生成 ====")
    panel_data = generate_test_panel_data(
        n_ids=2,
        n_treated=1,
        start_year=2010,
        pre_periods=8,
        post_periods=10,
        x_num=3,
        seed=123
    )
    print(panel_data)