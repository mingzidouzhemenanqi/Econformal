
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def get_treated_individuals(data, id, time, treat_col):
    """
    获取处理组所有个体ID，返回列表
    
    参数:
    data -- 包含面板数据的DataFrame
    id -- 个体ID列名 (string)
    time -- 时间列名 (string)
    treat_col -- 处理状态列名 (01变量，处理=1)
    
    返回:
    list -- 处理组个体的ID列表
    """
    # 筛选所有处理记录
    treated = data[data[treat_col] == 1]
    
    # 如果没有处理记录，返回空列表
    if treated.empty:
        return []
    
    # 获取处理组的所有个体ID
    treated_ids = treated[id].unique().tolist()
    
    return treated_ids



def get_first_treatment_year(data, id, time, treat_col):
    """
    获取面板数据中的首次处理年份（考虑多个处理个体）
    异常:
    - 如果没有处理记录存在，抛出ValueError
    """
    # 筛选所有处理状态为1的记录
    treated_data = data[data[treat_col] == 1]
    
    # 检查是否存在处理记录
    if treated_data.empty:
        raise ValueError("未找到任何处理记录：处理状态列中未发现值为1的记录")
    
    # 获取所有首次处理时间（每个个体的最小处理年份）
    first_treatments = (treated_data.groupby(id)
                        [time]
                        .min()
                        .reset_index(name='first_year'))
        
    # 返回所有首次处理时间中的最小值（即数据集中最早的处理年份）
    result = first_treatments['first_year'].min()
    if pd.isna(result):
        raise ValueError(
            "无法确定首次处理时间：处理记录中包含 NaN 时间值。"
            "请检查数据中处理组个体的时间列是否包含缺失值。"
        )
    return result


def extract_treatment_sc(data, id, time, treat_col):
    """
    提取处理组首次处理时间和处理组名称
    该函数为SC设计，因此会检查处理组样本数量是否为1
    
    参数:
    data : 数据表
    id : 样本列名
    time : 时间列名
    treat_col : 是否处理标识(0-1)
    
    返回:
    treatment_time : 首次处理年份
    treated_unit : 处理个体名称
    
    异常:
    - 如果没有处理组存在，抛出ValueError
    - 如果处理组包含多个个体，抛出ValueError
    """
    # 筛选处理组观察值（所有处理状态为1的记录）
    treated_obs = data[data[treat_col] == 1]
    
    # 检查是否存在处理组
    if treated_obs.empty:
        raise ValueError("No treatment group found: 处理状态列中未找到任何处理记录")
    
    # 获取所有处理过的个体ID
    treated_units = treated_obs[id].unique()
    
    # 检查处理个体数量
    if len(treated_units) > 1:
        raise ValueError(f"发现多个处理个体: {len(treated_units)}. 该计量算法要求处理组只有一个个体")
    
    # 提取唯一处理个体
    treated_unit = treated_units[0]
    
    # 获取该个体的首次处理年份
    treatment_time = treated_obs[treated_obs[id] == treated_unit][time].min()

    if pd.isna(treatment_time):
        raise ValueError(
            f"无法确定处理个体 '{treated_unit}' 的首次处理时间："
            f"该个体的处理记录中包含 NaN 时间值。"
        )

    # 返回首次处理时间和处理个体
    return treatment_time, treated_unit



def get_id_code(data, id_col):
    """
    给每个个体添加唯一数字ID编码，并赋值给id_code列
    """
    id_mapping = {id_val: idx + 1 for idx, id_val in enumerate(data[id_col].unique())}
    data['id_code'] = data[id_col].map(id_mapping)

    return data



def identify_panel_data(data, year_col, id_col, y_col, controls_col, treat_col):
    """
    面板数据识别函数
    该函数用于识别面板数据的处理组、处理时间，并新增一个id_code列，用于标识每个观测值。
    
    参数：
    data : DataFrame - 包含面板数据的DataFrame
    year_col : str   - 时间列名
    id_col : str     - 个体ID列名
    y_col : str      - 因变量列名
    controls_col : list    - 自变量列名列表
    treat_col : str  - 处理标识列名（0/1二值变量）
    
    返回：
    processed_data : 新增id_code列的预处理数据
    result_dict : 包含处理信息的字典
        - 'treat_groups': 处理组id列表
        - 'first_treat_dict': 按首次处理年份分组的处理组字典 {年份: [id列表]}
    """
    
    # 1. 基础验证
    required_cols = [year_col, id_col, y_col, treat_col] + controls_col
    missing = set(required_cols) - set(data.columns)
    if missing:
        raise ValueError(f"缺失必要列: {missing}")
    
    if data[treat_col].nunique() != 2 or set(data[treat_col].unique()) != {0, 1}:
        raise ValueError("处理列必须是0/1二值变量")
    
    # 复制数据避免修改原始数据
    data = data.copy()
    
    # 2. 添加唯一数字ID编码
    id_mapping = {id_val: idx + 1 for idx, id_val in enumerate(data[id_col].unique())}
    data['id_code'] = data[id_col].map(id_mapping)
    
    # 3. 标识处理组和控制组
    treat_groups = data.groupby(id_col)[treat_col].max()
    treat_group_ids = treat_groups[treat_groups == 1].index.tolist()
    
    # 4. 获取处理组首次处理年份（按年份分组）
    first_treat_data = data[data[treat_col] == 1].copy()
    
    # 找出每个处理组个体的首次处理年份
    first_treat_df = first_treat_data.groupby(id_col)[year_col].min().reset_index()
    
    # 创建按年份分组的字典
    year_grouped = defaultdict(list)
    for _, row in first_treat_df.iterrows():
        year_grouped[row[year_col]].append(row[id_col])
    
    # 将defaultdict转为常规dict
    first_treat_dict = dict(year_grouped)
    
    # 5. 创建返回字典
    result_dict = {
        'treat_groups': treat_group_ids,
        'first_treat_dict': first_treat_dict
    }
    
    return data, result_dict



# 检查数据是否为强面板数据，否则抛出警告
def strong_panel(data, id_col='id', time_col='time'):
    """
    检查数据集是否为强面板数据（平衡面板）
    
    参数：
        data: pandas DataFrame
        id_col: 个体标识列名 (默认'id')
        time_col: 时间标识列名 (默认'time')
    
    返回：
        bool: 是否为强面板数据
        str: 检查结果描述
    """
    # 1. 检查必需列是否存在
    if id_col not in data.columns:
        raise ValueError(f"列 '{id_col}' 不存在于数据集中")
    if time_col not in data.columns:
        raise ValueError(f"列 '{time_col}' 不存在于数据集中")
    
    # 2. 检查重复观测值 (每个id-time组合应唯一)
    duplicates = data.duplicated(subset=[id_col, time_col], keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        warning_msg = f"发现 {dup_count} 个重复的(id, time)观测值"
        warnings.warn(warning_msg, UserWarning)
        return False, warning_msg
    
    # 3. 检查平衡面板 (每个个体具有相同的时间点)
    # 获取所有唯一时间点
    all_times = sorted(data[time_col].unique())
    
    # 检查每个个体是否具有完整的时间序列
    grouped = data.groupby(id_col)[time_col]
    missing_info = []
    
    for entity_id, time_series in grouped:
        entity_times = sorted(time_series.unique())
        # 检查时间点是否完整
        if set(entity_times) != set(all_times):
            missing_times = set(all_times) - set(entity_times)
            missing_info.append(f"个体 {entity_id} 缺少时间点: {sorted(missing_times)}")
    
    # 4. 结果判断
    # 注：若 id→time 检查已确认每个个体具有所有时间点 (平衡面板)，
    # 则 time→id 的方向必然也完整，跳过冗余检查
    if missing_info:
        warning_msg = "非平衡面板数据:\n" + "\n".join(missing_info[:10])  # 最多显示10条信息
        if len(missing_info) > 10:
            warning_msg += f"\n...以及另外 {len(missing_info)-10} 条不平衡信息"
        warnings.warn(warning_msg, UserWarning)
        return False, warning_msg
    else:
        print('The data set is a strongly balanced panel data')
        return True, None

def validate_coverage(coverage):
    """校验 coverage 在 (0, 1) 范围内。

    Args:
        coverage: float，用户传入的目标覆盖率。

    Returns:
        bool: 校验通过返回 True。

    Raises:
        ValueError: coverage 不在 (0, 1) 区间内。
    """
    if not 0 < coverage < 1:
        raise ValueError(
            f"coverage 必须在 (0, 1) 范围内，当前值为 {coverage}。"
            f"例如: coverage=0.9 表示 90% 置信区间。"
        )
    return True


def data_col_name_check_T_(data):
    # 检查 data 每一列的列名是否以 'T_' 开头
    # 若包含，抛出错误，提示：列名不能以 'T_' 开头，否则影响 DID 等模型的估计
    for col in data.columns:
        if col.startswith('T_'):
            raise ValueError(f"列名不能以 'T_' 开头，否则影响 DID 等模型的估计。请检查列名：{col}")
    return True


def validate_treat_col(data, treat_col):
    """校验 treat_col 是否为 {0, 1} 二值整数变量且两种值均存在。

    Python 中 bool 是 int 的子类（True==1, False==0），float 的 0.0/1.0 也等于 0/1，
    因此仅靠 set(...) != {0, 1} 无法区分。本函数同时检查 dtype 确保为整数类型。

    Args:
        data: pd.DataFrame，输入面板数据。
        treat_col: str，处理指示列名。

    Raises:
        ValueError: 列不存在、值不是 {0, 1}、缺少 0/1 其中之一、或 dtype 非整数。
    """
    if treat_col not in data.columns:
        raise ValueError(f"处理列 '{treat_col}' 不在数据中，当前列名: {list(data.columns)}")

    col = data[treat_col]
    unique_vals = set(col.unique())
    # 值检查：必须恰好为 {0, 1}（int/bool/float 的 == 比较均适用）
    if unique_vals != {0, 1}:
        raise ValueError(
            f"处理列 '{treat_col}' 必须为 0/1 二值变量。"
            f"当前唯一值: {sorted(unique_vals)}，期望: [0, 1]"
        )
    # dtype 检查：拒绝 bool（True/False）和 float（0.0/1.0），要求整数
    if col.dtype == bool or col.dtype not in (int,):
        import numpy as np
        if col.dtype == bool:
            raise ValueError(
                f"处理列 '{treat_col}' 的 dtype 为 bool（True/False）。"
                f"请转换为 0/1 整数类型（如 data['{treat_col}'] = data['{treat_col}'].astype(int)）。"
            )
        if not np.issubdtype(col.dtype, np.integer):
            raise ValueError(
                f"处理列 '{treat_col}' 的 dtype 为 {col.dtype}，"
                f"必须为整数类型（0/1）。请使用 astype(int) 转换。"
            )
    return True


def validate_data_format(data, time_col, id_col, treat_col):
    """聚合数据格式校验：时间可排序、强面板、列名不含T_、treat列为0/1二值。"""
    validate_time_sortable(data, time_col)
    strong_panel(data, id_col=id_col, time_col=time_col)
    data_col_name_check_T_(data)
    validate_treat_col(data, treat_col)


def validate_y_col_and_controls(data, y_col, controls_col):
    """校验 y_col 和 controls_col 的列存在于 data 中，且不与其他关键列重叠。

    Args:
        data: pd.DataFrame
        y_col: str，因变量列名
        controls_col: list[str] 或 None，控制变量列名列表

    Raises:
        ValueError: 若列名不存在于 data 中，或 controls_col 与关键列重叠
    """
    if y_col not in data.columns:
        raise ValueError(
            f"因变量列 '{y_col}' 不在数据中，当前列名: {list(data.columns)}"
        )
    if controls_col:
        missing_ctrl = [c for c in controls_col if c not in data.columns]
        if missing_ctrl:
            raise ValueError(
                f"控制变量列 {missing_ctrl} 不在数据中，当前列名: {list(data.columns)}"
            )
        # G5: 检查 controls_col 不与 y_col/treat_col/time/id 重叠
        _reserved = {y_col}
        for c in controls_col:
            if c in _reserved:
                raise ValueError(
                    f"控制变量列 '{c}' 与因变量列 '{y_col}' 重名。"
                    f"控制变量不能包含因变量列。"
                )


def validate_no_nan(data, y_col, treat_col, time_col, id_col, controls_col):
    """校验关键列中不存在 NaN 值。

    Args:
        data: pd.DataFrame
        y_col, treat_col, time_col, id_col: str，关键列名
        controls_col: list[str] 或 None，控制变量列名列表

    Raises:
        ValueError: 若任何关键列包含 NaN
    """
    _critical = [y_col, treat_col, time_col, id_col]
    if controls_col:
        _critical.extend(controls_col)
    _critical = [c for c in _critical if c in data.columns]  # 仅检查存在的列
    nan_cols = [c for c in _critical if data[c].isna().any()]
    if nan_cols:
        raise ValueError(
            f"以下关键列包含 NaN 值: {nan_cols}。"
            f"请先处理缺失值后再进行分析（如 dropna 或 fillna）。"
        )


def preprocess_data(data, id_col, time_col):
    """数据预处理：生成样本识别码 id_code，按时间、个体排序。"""
    data = get_id_code(data, id_col)
    data = data.sort_values(by=[time_col, id_col]).reset_index(drop=True)
    return data


def validate_time_sortable(data, time_col):
    """校验时间列是否可排序。

    Args:
        data: pd.DataFrame，输入面板数据。
        time_col: str，时间列名。

    Raises:
        ValueError: 列不存在，或列值类型不可排序。
    """
    if time_col not in data.columns:
        raise ValueError(f"时间列 '{time_col}' 不在数据中，当前列名: {list(data.columns)}")

    try:
        sorted(data[time_col].unique())
    except TypeError as e:
        raise ValueError(
            f"时间列 '{time_col}' 的值类型不可排序（{e}）。"
            f"请确保时间列为 int、float、datetime 或可比较的字符串类型。"
        )
    return True


# 当用户选择SC时，检查treat_time和target_id
def data_sc(data: pd.DataFrame, time: str, id: str, treat_time, target_id):
    '''
    1. 检查treat_time是否为None以及以及是否在data的time列中
    2. 检查target_id是否为None以及是否在data的id列中
    '''

    'treat_time检查'
    if treat_time is None:
        raise ValueError("请输入处理时间不能为空")
    if treat_time not in data[time].unique():
        raise ValueError(f"处理时间 {treat_time} 不在数据{time}列中")

    'target_id检查'
    if target_id is None:
        raise ValueError("请输入目标个体不能为空")
    if target_id not in data[id].unique():
        raise ValueError(f"目标个体 {target_id} 不在数据{id}列中")

    return True