# Econformal

计量经济学模型共形推断工具包 | Conformal Inference Toolkit for Econometric Models

[![PyPI version](https://badge.fury.io/py/econformal.svg)](https://badge.fury.io/py/econformal)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 简介

Econformal 是一个将共形推断（Conformal Inference）与计量经济学模型相结合的 Python 工具包，为 DID、Synthetic Control 等计量模型的 treatment effect 提供具有统计保证的不确定性量化（置信区间）。

## 安装

```bash
pip install econformal
```

## 快速开始

以下四个示例从零开始，逐步展示 Econformal 的核心用法。每个代码块均可独立运行。

### 1. 认识面板数据格式

在使用任何计量模型之前，先了解 Econformal 需要的面板数据结构。

```python
from econformal.tools.generate_data import generate_test_panel_data

# 生成一个微型面板数据集（5 个个体，2 个处理组，6 个时期）
data = generate_test_panel_data(
    n_ids=5, n_treated=2, start_year=2000,
    pre_periods=4, post_periods=2, x_num=1, seed=99
)

print(data.head(12))
```

**列名含义：**

| 列名 | 说明 |
|------|------|
| `year` | 时间变量，需可排序（int/float/datetime） |
| `id` | 个体标识，每个个体有唯一 ID |
| `Y` | 因变量/结果变量（outcome） |
| `Treat` | 处理指示变量，**必须为 0/1 二值**。0=对照组（始终不接受处理），1=处理组（在处理时点之后接受处理） |
| `X1`, `X2`, ... | 协变量/控制变量（可选），用于控制混杂因素 |

**数据约束：**
- 必须是强面板（平衡面板）：每个 `(id, time)` 组合唯一
- `Treat` 列必须同时包含 0 和 1
- 列名不能包含 `'T_'` 前缀（DID 内部保留字）
- 无缺失值（NaN）

---

### 2. SC + Full Conformal —— 最严谨的组合

合成控制法适用于只有 1 个（或极少数）处理单元的场景。Full Conformal 通过置换检验在 nulls 网格上搜索置信区间，提供精确的有限样本覆盖保证。

```python
import numpy as np
import matplotlib.pyplot as plt
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data

# 1. 生成数据：30 个个体，仅 1 个处理单元
data = generate_test_panel_data(
    n_ids=30, n_treated=1, start_year=2000,
    pre_periods=10, post_periods=5, x_num=0, seed=99
)

# 2. 初始化（SC 不使用协变量，controls_col=None）
model = Econformal(
    data=data, time='year', id='id',
    y_col='Y', treat_col='Treat', controls_col=None
)

# 3. 设定原假设网格
# nulls 是一组候选处理效应值。Full Conformal 对每个值做置换检验，
# 保留 p-value ≥ (1-coverage) 的值，取 min/max 作为置信区间。
# 范围太窄 → 所有 null 被拒绝 → 边界为 NaN；范围太宽 → 计算量增大。
nulls = np.linspace(-10, 10, 50)

# 4. 执行共形推断（SC + Full Conformal，90% 置信水平）
result = model.conformal_inference(
    econ_model='sc', conformal_model='full',
    nulls=nulls, coverage=0.9
)
print(result.round(4))

# 5. 绘制置信区间图
fig = model.plot_ci_interval()
fig.savefig('sc_full_conformal.png', dpi=150, bbox_inches='tight')
plt.close()
```

**输出列说明：**

| 列名 | 说明 |
|------|------|
| `year` | 时间 |
| `effect` | 处理效应估计值（实际 Y − SC 合成预测 Y） |
| `predictions` | SC 合成预测值 |
| `std_error` | 标准误 |
| `p-value` | P 值 |
| `90%_conformal_lower` | 共形推断 90% CI 下界（仅处理后时期有值） |
| `90%_conformal_upper` | 共形推断 90% CI 上界 |

> ⚠ Full Conformal 的拟合次数 = 处理后时期数 × nulls 个数，本例约 250 次 SC 拟合，需数十秒。

---

### 3. DID + Split Conformal —— 快速版，适合大数据

双重差分法是应用最广泛的因果推断方法之一。Split Conformal 将处理前时期切分为训练集和校准集，仅需 **2 次**模型拟合，秒级出结果。

```python
import numpy as np
import matplotlib.pyplot as plt
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data

# 1. 生成含协变量的大样本数据
data = generate_test_panel_data(
    n_ids=200, n_treated=50, start_year=2010,
    pre_periods=15, post_periods=8, x_num=3, seed=123
)

# 2. 初始化（纳入协变量控制混杂因素）
model = Econformal(
    data=data, time='year', id='id', y_col='Y',
    treat_col='Treat', controls_col=['X1', 'X2', 'X3']
)

# 3. 执行共形推断
# split_rate=0.7: 前 70% 处理前时期用于训练，后 30% 用于校准分位数
# Split Conformal 不需要 nulls 参数
result = model.conformal_inference(
    econ_model='did', conformal_model='split',
    split_rate=0.7, coverage=0.90
)
print(result.round(4))

# 4. 绘制对比图：共形 CI（I 形线）vs 传统 CI（灰色阴影）
fig = model.plot_ci_interval(traditional=True)
fig.savefig('did_split_conformal.png', dpi=150, bbox_inches='tight')
plt.close()
```

**结果解读：**
- 处理前（竖虚线左侧）：effect 应接近 0（平行趋势假设）
- 处理后（竖虚线右侧）：effect 偏离 0 表示处理效应
- 共形 CI（I 形竖线）通常比传统 CI（灰色阴影）更宽 —— 更诚实地反映不确定性
- 传统 CI 依赖模型假设（正态性），共形 CI 提供分布无关的覆盖保证

---

### 4. SDID + CV+ Conformal —— 现代方法组合

合成双重差分（Arkhangelsky et al., 2021）结合了 DID 和 SC 的优势。CV+ 是 Jackknife+ 的 K 折泛化，将拟合次数从 n_pre 降至 K（默认 5）。

```python
import numpy as np
import matplotlib.pyplot as plt
from econformal import Econformal
from econformal.tools.generate_data import generate_test_panel_data

# 1. 生成数据：100 个个体，30 个处理单元（SDID 支持多处理单元）
data = generate_test_panel_data(
    n_ids=100, n_treated=30, start_year=2010,
    pre_periods=12, post_periods=6, x_num=2, seed=42
)

# 2. 初始化
model = Econformal(
    data=data, time='year', id='id', y_col='Y',
    treat_col='Treat', controls_col=['X1', 'X2']
)

# 3. 执行共形推断（SDID + CV+）
# cv_folds=5      : K=5 折交叉验证
# cv_strategy='block' : 按时序连续分块（保留时间依赖，推荐面板数据使用）
#                       'random' 为随机打乱分折（假设可交换性）
result = model.conformal_inference(
    econ_model='sdid', conformal_model='cv_plus',
    coverage=0.90, cv_folds=5, cv_strategy='block', random_state=42
)
print(result.round(4))

# 4. 绘制 CI 图
fig = model.plot_ci_interval(traditional=True)
fig.savefig('sdid_cvplus_conformal.png', dpi=150, bbox_inches='tight')
plt.close()
```

> ⚠ 方法论提醒：DID + JK+/CV+ 使用代理残差（非标准 held-out 预测误差），覆盖保证不严格成立。推荐 SC/SDID 搭配 JK+/CV+ 使用。

---

## 方法选择指南

三种计量模型 × 五种共形方法 = 15 种组合。按以下场景选择：

### 计量模型对比

| 模型 | 适用场景 | 处理单元数 | 协变量支持 | 核心思想 |
|------|---------|-----------|-----------|---------|
| **SC** | 单个或极少数处理单元 | 1 | 有限 | 对照组凸组合构造反事实 |
| **DID** | 大规模面板，多处理单元 | 任意 | ✓ | 双重差分 + 事件研究法 |
| **SDID** | 中等规模，追求稳健 | 多个 | ✓ | SC 权重 + DID 时间权重 |

### 共形方法对比

| 方法 | 拟合次数 | 区间形状 | 覆盖保证 | 适用场景 |
|------|---------|---------|---------|---------|
| **Full** | n_post × n_nulls | 各时点可不同 | 精确有限样本 | 小样本，追求最大严谨性 |
| **Split** | 2 | 各时点等宽 | 近似 | 大数据，快速迭代预览 |
| **LOO** | n_pre | 各时点等宽 | 仿真偏保守 | 中等数据，追求稳健 |
| **JK+** | n_pre | 各时点非对称 | P(Y∈CI) ≥ 1−2α | 中等数据，异方差场景 |
| **CV+** | K (默认 5) | 各时点非对称 | P(Y∈CI) ≥ 1−2α (近似) | 多时点，追求效率 |

### 推荐组合

| 场景 | 推荐组合 | 理由 |
|------|---------|------|
| 小样本，追求严谨 | SC + Full | SC 仅 1 个处理单元，Full 提供精确有限样本保证 |
| 大数据，快速迭代 | DID + Split | 2 次拟合，秒级出结果 |
| 常规分析，均衡高效 | SDID + CV+ | 现代方法 + K 折校准，适合正式报告 |
| 追求理论保证 | SC + JK+ / CV+ | SC 可做真正的 leave-one-out 预测 |

---

## 实用技巧

### 调参建议

- **nulls 范围太窄** → 置信区间边界为 NaN → 扩大 `np.linspace` 范围
- **Full Conformal 太慢** → 减少 nulls 点数（如 100 → 30）或换用 split
- **Split 区间太宽** → 增加 `split_rate`（更多训练数据改善模型拟合）
- **CV+ 折数选择** → 处理前时期少用 `block` + 少折；时期多用 `random` + 多折
- **coverage 选择** → 90% 为常用默认，95% 更保守，80% 更窄但覆盖更低

### 使用自己的数据

```python
import pandas as pd
import numpy as np
from econformal import Econformal

my_data = pd.read_csv('my_panel_data.csv')
model = Econformal(
    data=my_data,
    time='year',           # 你的时间列名
    id='state',            # 你的个体列名
    y_col='outcome',       # 你的因变量列名
    treat_col='treated',   # 你的处理变量列名（必须 0/1）
    controls_col=['gdp', 'population']  # 可选协变量
)
result = model.conformal_inference(
    econ_model='did',
    conformal_model='full',
    nulls=np.linspace(-20, 20, 100),
    coverage=0.95
)
print(result)
```

### 保存高清图片

```python
fig = model.plot_ci_interval(traditional=True)
fig.savefig('my_ci_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## 核心模块

- **conformal_methods**: 共形推断方法
  - `full` — Full Conformal（原假设网格搜索 + 置换检验）
  - `split` — Split Conformal（训练/校准集切分 + 分位数）
  - `loo` — Leave-One-Out Conformal（留一法校准）
  - `jk_plus` — Jackknife+ Conformal（非对称区间）
  - `cv_plus` — CV+ Conformal（K 折交叉验证）

- **econometrics_methods**: 计量经济学方法
  - `did` — 双重差分（事件研究法 + PanelOLS）
  - `sc` — 合成控制（cvxpy 凸优化求解权重）
  - `sdid` — 合成双重差分（单元/时间双重加权）

- **tools**: 工具函数
  - `check` — 数据校验（强面板检查、列名检查、缺失值检查）
  - `generate_data` — 模拟面板数据生成
  - `plot` — 可视化（效应曲线 + 共形置信区间）
  - `model_registration` — 动态模型加载

## 示例脚本

`example.py` 提供了可直接运行的 3 段式简单示例（SC/DID/SDID + Full Conformal）。

## 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证
本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。


