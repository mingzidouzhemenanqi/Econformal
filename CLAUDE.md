# Econformal v0.1.0 — 计量经济学共形推断工具包

Econformal 是一个将共形推断（Conformal Inference）与计量经济学模型相结合的 Python 工具包，为 DID、Synthetic Control 等计量模型的 treatment effect 提供具有统计保证的不确定性量化（置信区间）。

## 项目结构

```
Econformal/
├── pyproject.toml          # 项目配置（hatchling 构建，依赖声明）
├── example.py              # 使用示例（DID/SC/SDID）
├── src/econformal/
│   ├── __init__.py         # 包入口，导出 Econformal 类
│   ├── base.py             # Econformal 主类：用户 API、数据校验、模型调度、结果合并
│   ├── conformal_methods/
│   │   ├── conformal_base.py  # ConformalBase 抽象基类
│   │   ├── full.py            # Full Conformal：遍历 nulls 网格搜索置信区间
│   │   └── split.py           # Split Conformal：训练/校准集切分 + 分位数
│   ├── econometrics_methods/
│   │   ├── did.py             # DID（事件研究法），基于 PanelOLS + 相对时间虚拟变量
│   │   └── sc.py              # Synthetic Control，基于 cvxpy 凸优化求解权重
│   │   └── sdid.py            # Synthetic DID (Arkhangelsky et al., 2021)，单元/时间双重加权
│   └── tools/
│       ├── check.py           # 数据校验：强面板检查、列名检查、处理组/时间提取
│       ├── plot.py            # 可视化：效应曲线 + 共形置信区间
│       ├── generate_data.py   # 模拟面板数据生成
│       └── model_registration.py  # 动态模型加载（按字符串名导入模块）
├── design/                # 设计文档
├── ignore/                # 测试数据
└── .claude/               # Claude Code 配置
```

## 环境信息

- **Python**: 3.14（`.python-version`）
- **包管理**: uv（`uv.lock` 已提交）
- **构建**: hatchling
- **PyPI**: `econformal` (v0.1.0)
- **操作系统**: Windows 11

### 重要：uv 命令使用规范

本项目采用 uv 管理虚拟环境，**所有 Python 命令和包安装必须通过 uv 执行**，否则可能因环境不一致而报错：

```bash
# 运行脚本
uv run python script.py

# 单行代码
uv run python -c "print('hello')"

# 安装依赖
uv add <package>

# 安装开发依赖
uv add --dev <package>

# 同步依赖（安装 uv.lock 中的所有包）
uv sync
```

## 核心架构

### 用户 API 入口

```python
from econformal import Econformal

model = Econformal(data=data, time='year', id='id', y_col='Y',
                   treat_col='Treat', controls_col=['X1', 'X2'])

result = model.conformal_inference(
    econ_model='did',        # 计量模型: 'did' | 'sc' | 'sdid'
    conformal_model='full',  # 共形方法: 'full' | 'split'
    nulls=np.linspace(-10, 10, 100),
    coverage=0.90,
)

fig = model.plot_ci_interval()  # 或 plot_ci_interval(traditional=True)
```

### 动态模型加载

`model_registration.py` 根据字符串名动态导入模块：
- `get_econ_model('did')` → 导入 `econometrics_methods.did` → 返回 `Econometric` 类
- `get_conformal_model('full')` → 导入 `conformal_methods.full` → 返回 `Conformal` 类

每个计量/共形模块的类名固定：计量模型类名为 `Econometric`，共形模型类名为 `Conformal`。

### 计量模型接口

所有计量模型类必须实现 `fit_econmodel(data, time, id, y_col, treat_col, coverage, controls_col=None, **kwargs)` 方法，返回 DataFrame（含 effect、std_error、p-value、置信区间列）。

### 共形模型接口

所有共形模型继承 `ConformalBase`，必须实现:
- `compute_conformal_interval()` — 主入口
- `fit()` — 计算核心统计量
- `predict()` — 生成预测区间
- `preprocess_data()` — 数据预处理
- `result_to_dataframe()` — 结果格式化

### 数据流

```
DataFrame 输入 → Econformal.__init__() → 数据校验/排序/编码
→ conformal_inference()
  → _econ_fit() → 计量模型拟合 → self.econ_results
  → _conformal_inference_fit() → 共形推断 → self.conformal_interval
  → _merge_results() → 按 time 列 outer join → self.results
→ plot_ci_interval() → 可视化
```

## 数据结构全链路

### 1. 输入数据结构（`Econformal.__init__`）

用户传入的 `pd.DataFrame`，列名由用户自定义：

| 参数 | 类型 | 说明 |
|--------|------|------|
| `time` | str (列名) | 时间变量，需可排序（int/float/datetime），如 `'year'` |
| `id` | str (列名) | 个体标识，如 `'id'` |
| `y_col` | str (列名) | 因变量，数值型 |
| `treat_col` | str (列名) | 处理指示变量，**必须为 0/1 二值**，且 0 和 1 均需出现 |
| `controls_col` | list[str] \| None | 控制变量列名列表，可选，默认为 `None` |

**数据约束**：
- 必须是强面板（平衡面板）：每个 `(id, time)` 组合唯一，所有个体覆盖相同时段
- 列名**不能**包含 `'T_'` 前缀（DID 内部会生成 `event_{t}` 列，`T_` 是 PanelOLS 虚拟变量前缀）
- 时间列必须可排序，否则无法计算相对事件时间

示例：
```python
# 列: year, id, Y, Treat, X1, X2
data = generate_test_panel_data(n_ids=100, pre_periods=3, post_periods=2, x_num=2)
Econformal(data, time='year', id='id', y_col='Y', treat_col='Treat', controls_col=['X1', 'X2'])
```

### 2. 初始化后数据结构（`self.data`）

`Econformal.__init__` 完成校验和预处理后，`self.data` 在原始 DataFrame 基础上：

- 新增 `id_code` 列：个体的数字编码（从 1 开始，按 `id` 列 unique 值映射）
- 按 `(time, id)` 排序
- 保留原始的所有列不变

### 3. 计量模型输出（`self.econ_results`）

`_econ_fit()` → 各计量模型的 `fit_econmodel()` 返回，结构因模型而异：

#### DID 输出

| 列名 | 类型 | 说明 |
|------|------|------|
| `{time}` | 与输入同类型 | 原始时间值（从相对时间映射回） |
| `effect` | float64 | 处理效应估计值 |
| `std_error` | float64 | 聚类稳健标准误（cluster by entity） |
| `p-value` | float64 | P 值 |
| `{cov}%_conf_lower` | float64 | 传统置信区间下界 |
| `{cov}%_conf_upper` | float64 | 传统置信区间上界 |

> DID 内部流程：`原始时间 → relative_time（以首次处理期为 0）→ 创建 event_{t} 虚拟变量 → PanelOLS(entity_effects=True, time_effects=True) → 提取系数 → relative_time 映射回原始时间`

> 基准期（t=-1）被省略以避免多重共线性，其 effect=0, std_error=0, p-value=1.0

#### SC 输出

| 列名 | 类型 | 说明 |
|------|------|------|
| `{time}` | int | 原始时间值 |
| `predictions` | float64 | SC 合成预测值（`X @ w`） |
| `effect` | float64 | 实际值 - 预测值 |
| `std_error` | float64/0 | 处理前残差的标准差（ddof=1），处理后为 0 |
| `p-value` | 0 | 占位（SC 尚未实现 p-value 计算） |
| `置信区间下界` | float64/0 | 正态近似 CI 下界（处理前残差 ±z 值） |
| `置信区间上界` | float64/0 | 正态近似 CI 上界 |

> SC 内部流程：`去除控制变量 → 长变宽 pivot(id, time) → 仅用处理前数据 cvxpy 求解权重 w（sum(w)=1, w≥0）→ 全时段预测 → effect = 实际 - 预测`

> ⚠️ SC 传统 CI 列名为硬编码中文，与 DID 的 `{cov}%_conf_lower/upper` 命名不一致

### 4. 共形推断中间过程

#### Full Conformal

**`preprocess_data()` → `time_list` + `p_value_matrix`:**
- `time_list`: 处理后时期的时间值数组，shape `(n_post, )`
- `p_value_matrix`: 零矩阵，shape `(n_post, len(nulls))`

**`fit()` 填充 `p_value_matrix`:**
- 双层循环：外层遍历 `time_list`，内层遍历 `nulls`
- 对每个 `(t, null)`：在 `t` 时刻给处理组 Y 减去 `null`（构造增强数据集）→ 重新拟合计量模型 → 提取 effect 残差 → 计算置换 p-value
- `self.p_value_matrix[i, j]` = 在时间 `t_i`、原假设 `null_j` 下的 p 值

**`predict()` → `confidence_interval`:**
- 将 `p_value_matrix < (1-coverage)` 对应的 nulls 置为 NaN
- 每行取 `(nanmin, nanmax)` 作为该时间点的共形置信区间
- 输出 shape `(n_post, 2)` 的 numpy array

**`result_to_dataframe()` → `self.conformal_interval`:**
- index = `time_list`（处理后时间）
- columns = `['{cov}%_conformal_lower', '{cov}%_conformal_upper']`

#### Split Conformal

**`fit()` 内部:**
- 按 `splite_rate`（默认 70%）将处理前时间分为训练集和校准集
- 训练集拟合计量模型 → 校准集计算残差 → 取 `ceil((n+1)*coverage)` 分位数作为 `self.quantile`

**`predict()` → `self.conformal_interval`:**
- 用训练集+处理后数据重新拟合 → 提取处理后残差
- 区间 = `residuals ± quantile`
- 输出 DataFrame，index = 处理后时间，columns = `['{cov}%_conformal_lower', '{cov}%_conformal_upper']`

### 5. 最终合并输出（`self.results`）

`_merge_results()` 将 `self.conformal_interval` 和 `self.econ_results` 按时间列 **outer join**：

| 列名 | 来源 | 说明 |
|------|------|------|
| `{time}` | 两者 | 原始时间（int/float/str，合并前统一类型） |
| `{cov}%_conformal_lower` | conformal_interval | 共形推断下界（仅处理后时期有值） |
| `{cov}%_conformal_upper` | conformal_interval | 共形推断上界（仅处理后时期有值） |
| `predictions` | econ_results（仅 SC） | SC 预测值，DID 此列为 NaN |
| `effect` | econ_results | 处理效应 |
| `std_error` | econ_results | 标准误 |
| `p-value` | econ_results | P 值 |
| `{cov}%_conf_lower` | econ_results（DID） | 传统 CI 下界 |
| `{cov}%_conf_upper` | econ_results（DID） | 传统 CI 上界 |
| `置信区间下界` | econ_results（SC） | 传统 CI 下界（中文名） |
| `置信区间上界` | econ_results（SC） | 传统 CI 上界（中文名） |

> 合并前会校验：重复值、NaN、空 DataFrame。

### 6. 可视化输出

`plot_ci_interval()` 使用 `self.results` 生成 matplotlib 图：
- 以 `{time}` 为 x 轴，`effect` 为 y 轴折线
- 着色区域：`[{cov}%_conformal_lower, {cov}%_conformal_upper]`
- 参考线：y=0（横轴）、x=treat_time（首次处理时间竖线）
- `traditional=True` 时叠加传统 CI 误差棒

### 数据流总览图

```
用户 DataFrame
  │
  ├─ __init__: strong_panel检查 → get_id_code → 排序 → self.data
  │
  └─ conformal_inference()
       │
       ├─ _econ_fit()
       │    ├─ DID: _calculate_relative_time → _create_event_dummies → PanelOLS → self.econ_results
       │    └─ SC:  pivot宽表 → cvxpy求解w → predict → effect → self.econ_results
       │
       ├─ _conformal_inference_fit()
       │    ├─ Full: 双层循环(nulls×time) → p_value_matrix → predict(nanmin/nanmax) → self.conformal_interval
       │    └─ Split: 切分训练/校准集 → 残差分位数 → residuals±quantile → self.conformal_interval
       │
       ├─ _merge_results(): outer join on time → self.results
       │
       └─ plot_ci_interval(): matplotlib 折线+着色区间
```

## 关键依赖

- `statsmodels`, `linearmodels` — DID 面板回归
- `cvxpy` — SC 权重凸优化
- `pandas`, `numpy` — 数据处理
- `matplotlib`, `seaborn` — 可视化
- `scikit-learn` — SC 模型基类（`BaseEstimator`）
- `tqdm` — Full Conformal 进度条

## 项目状态

- 当前版本 0.1.0（Alpha），已发布 PyPI
- 已实现：DID (事件研究法)、SC (合成控制)、SDID (合成双重差分)、Full/Split/LOO/JK+/CV+ Conformal
- SC 控制变量支持有限（增强矩阵法，需至少一个处理后时期）
- SDID 支持控制变量（增强矩阵法，同 SC 模式），支持多处理单元
- Split/LOO/JK+/CV+ Conformal 已接入 `ConformalBase` 抽象基类
