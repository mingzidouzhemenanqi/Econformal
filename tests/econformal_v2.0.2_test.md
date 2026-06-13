# Econformal v0.2.0 测试报告

> **生成日期**: 2026-06-12  
> **测试环境**: Windows 11 Pro, Python 3.14.5, pytest 9.0.3  
> **分支**: `check_all` (基于 `main` @ `102ef1f`)

---

## 一、项目概况

Econformal 是一个将共形推断（Conformal Inference）与计量经济学模型相结合的 Python 工具包，为 DID、SC、SDID 等计量模型的 treatment effect 提供具有统计保证的不确定性量化。

| 项目 | 详情 |
|------|------|
| 版本 | 0.2.0 |
| Python | >=3.8, 测试于 3.14.5 |
| 包管理 | uv + hatchling |
| 核心依赖 | pandas, numpy, statsmodels, linearmodels, cvxpy, scikit-learn, matplotlib, tqdm |
| 测试框架 | pytest 9.0.3 |

---

## 二、测试架构

```
tests/
├── conftest.py                     # 共享 fixtures (2 数据生成器)
├── test_combinations.py            # 31 测试 — 15 组合 + 边界用例
├── test_econometric_correctness.py # 7 测试 — 计量模型数学正确性
├── test_conformal_correctness.py   # 15 测试 — 共形模型数学正确性
├── test_final_integration.py       # 27 测试 — 全面集成验证
└── econformal_v2.0.2_test.md       # 本报告
```

**总计: 80 项测试, 0 失败, 0 错误**

---

## 三、15 种模型组合测试矩阵

所有 3 种计量模型 × 5 种共形模型的组合均已完成端到端验证。每个组合均经过至少两个测试文件的校验。

| # | 计量模型 | 共形方法 | test_combinations | test_final_integration | 状态 |
|---|---------|---------|:---:|:---:|:---:|
| 1 | DID | Full | ✅ | ✅ | PASS |
| 2 | DID | Split | ✅ | ✅ | PASS |
| 3 | DID | LOO | ✅ | ✅ | PASS |
| 4 | DID | JK+ | ✅ | ✅ | PASS |
| 5 | DID | CV+ | ✅ | ✅ | PASS |
| 6 | SC | Full | ✅ | ✅ | PASS |
| 7 | SC | Split | ✅ | ✅ | PASS |
| 8 | SC | LOO | ✅ | ✅ | PASS |
| 9 | SC | JK+ | ✅ | ✅ | PASS |
| 10 | SC | CV+ | ✅ | ✅ | PASS |
| 11 | SDID | Full | ✅ | ✅ | PASS |
| 12 | SDID | Split | ✅ | ✅ | PASS |
| 13 | SDID | LOO | ✅ | ✅ | PASS |
| 14 | SDID | JK+ | ✅ | ✅ | PASS |
| 15 | SDID | CV+ | ✅ | ✅ | PASS |

每个组合均验证：
- API 调用无崩溃
- 返回非空 DataFrame
- 包含 `effect`、`{cov}%_conformal_lower`、`{cov}%_conformal_upper` 列
- 处理后时期的 CI 下界 ≤ CI 上界
- effect 列无 NaN

---

## 四、测试分类明细

### 4.1 接口与参数测试 (test_combinations.py — 31 tests)

| 测试 | 数量 | 描述 |
|------|:---:|------|
| 15 组合参数化测试 | 15 | 每种组合的端到端验证 |
| 无控制变量测试 | 3 | `controls_col=[]` 对 DID/SC/SDID |
| 无效输入测试 | 7 | 无效模型名、coverage 越界、缺失列、处理前期不足 |
| kwargs 透传测试 | 4 | event_window、zeta、random_state、未识别参数 |
| 未识别 kwargs 警告 | 1 | 拼写错误检测 |
| CI 结构验证 | 1 | CI 边界有效性 |

### 4.2 计量模型正确性 (test_econometric_correctness.py — 7 tests)

| 测试 | 描述 |
|------|------|
| `test_did_all_treated_raises` | 全部个体为处理组 → ValueError |
| `test_did_staggered_adoption_raises` | 多个处理时间 → ValueError |
| `test_did_missing_event_rows_become_nan` | event_window 超出数据范围 → NaN 行 |
| `test_sc_ci_centered_on_effect` | SC CI 以效应估计值为中心 |
| `test_sc_unbalanced_panel_raises` | 不平衡面板 → ValueError |
| `test_sdid_zeta_pooled` | SDID zeta 使用合并标准差 |
| `test_sdid_unit_weights_correctness` | SDID 权重正确性 |

### 4.3 共形方法正确性 (test_conformal_correctness.py — 15 tests)

| 测试 | 描述 |
|------|------|
| Full Conformal 验证 | ×3 (DID/SC/SDID) |
| JK+ 索引保护 | 小样本下界钳制 |
| SC+JK+ 残差 | held-out 残差正确性 |
| CV+ 策略 | random + block 两种策略 |
| LOO 全模型 | ×3 (DID/SC/SDID) |
| CI 非零宽度 | ×5 组合 |

### 4.4 最终集成测试 (test_final_integration.py — 27 tests)

| 测试 | 数量 | 描述 |
|------|:---:|------|
| 15 组合全面测试 | 15 | 完整验证 |
| 可复现性 | 1 | 相同种子 → 相同结果 |
| 多次调用 | 1 | 同一实例两次调用 |
| 自定义参数 | 3 | event_window, zeta, cv_folds |
| 无控制变量 | 3 | DID/SC/SDID |
| 不同覆盖率 | 4 | 80%, 90%, 95%, 99% |

---

## 五、六轮审计修复汇总

### Round 1 (22 fixes) — 基础 Bug 与数据流
- **严重 6 项**: DID event_window 存储、空数组 crash、SC/SDID merge 类型、Split 索引越界、Full NaN CI、SDID 模块名误判
- **高 7 项**: Full+DID 时间映射、列存在性验证、pre_times 下限、SC p-value/std_error、cvxpy 状态、SDID seed、placebo 缓存
- **中 8 项**: 置换检验文档、DID 基准期、共线性文档、kwargs 吸收、T_ 前缀检查、generate_data 向量化、强面板冗余、SC std ddof
- **低 4 项**: 文档字符串、函数名拼写、可变默认参数、模型注册大小写

### Round 2 (8 fixes) — Kwargs 透传与时间类型
- **严重 3 项**: F1 kwargs 透传链、F3 SC/SDID astype、F6 DID 动态省略基准期
- **高 2 项**: F4 SC 残差修剪、F5 DID event_window 超界过滤
- **中 1 项**: F7 SC DataFrame 构造
- **低 2 项**: DID 全零虚拟变量过滤、event_window 缓存回退

### Round 3 (10 fixes) — 边界条件
- **严重 3 项**: G1 NaN 校验、G2 treat_col dtype、G3 DID 结果循环
- **高 3 项**: G4 treat_time NaN、G5 controls_col 重叠、G6 自适应容差
- **中 4 项**: G7 可变默认值、G8 孤儿实例、G9 Full predict 守卫、G10 RNG 隔离

### Round 4 (6 fixes) — 计量模型数学正确性
- **高 4 项**: D1 DID 静默行丢弃、D2 全部处理组检查、D3 SC CI 居中、D4 交错处理检测
- **中 2 项**: D5 SC pivot NaN、D6 SDID zeta 合并标准差

### Round 5 (4 fixes) — 共形方法数学正确性
- **高 2 项**: E1 JK+ 索引钳制、E2 Full 验证调用
- **中 1 项**: E4 LOO 死代码移除
- **低 1 项**: cv_folds/strategy 已知参数

**总计: ~50 项修复**

---

## 六、已知限制与注意事项

### 6.1 DID 模型
- 仅支持单一处理时间点（同时处理），不支持交错处理
- 相对时间基于位置计数（非日历距离），非连续时间序列需注意
- 需要至少 1 个从未处理的对照个体
- `event_window` 默认自动检测，用户自定义窗口需确认在数据范围内

### 6.2 SC 模型
- 仅支持 1 个处理个体
- 传统 CI 基于正态近似（建议使用共形推断 CI 替代）
- p-value 不可用（返回 NaN）
- 需要强平衡面板

### 6.3 SDID 模型
- 需要至少 2 个预处理时期 + 1 个处理后期
- 需要至少 1 个对照个体（建议 >= 5 以获得可靠推断）
- 控制变量增强为实验性功能（与论文方法不同）
- Full Conformal 模式下 placebo 标准误可能来自不同数据规模

### 6.4 共形方法
- **JK+/CV+ 与 DID 组合**: R_j 使用代理残差（非标准 held-out 残差），未保证理论上的 1-2α 覆盖
- **Full Conformal**: 置换检验使用圆形移位（非块置换），有趋势数据可能产生偏差
- **Split Conformal**: 小校准集（n < 10）下覆盖保证可能不足
- **CV+**: 随机折划分（默认策略）不保持时间序列结构

---

## 七、性能基准

| 测试套件 | 测试数 | 耗时 |
|---------|:---:|------|
| test_combinations.py | 31 | ~11s |
| test_econometric_correctness.py | 7 | ~2s |
| test_conformal_correctness.py | 15 | ~7s |
| test_final_integration.py | 27 | ~15s |
| **总计** | **80** | **~35s** |

SDID+Full Conformal 是最耗时的组合（placebo 计算 O(N_co) 次 SDID 拟合）。

---

## 八、测试执行命令

```bash
# 运行全部测试
uv run pytest tests/ -v

# 运行特定套件
uv run pytest tests/test_combinations.py -v
uv run pytest tests/test_econometric_correctness.py -v
uv run pytest tests/test_conformal_correctness.py -v
uv run pytest tests/test_final_integration.py -v

# 运行特定组合测试
uv run pytest tests/ -k "did and full" -v
uv run pytest tests/ -k "sdid and jk" -v
```

---

## 九、结论

经过六轮深度审计和测试，Econformal v2.0.2 的：

1. **接口稳定性**: 所有 15 种模型组合均可通过 `Econformal(...).conformal_inference(...)` 正确调用
2. **输出正确性**: 所有组合均返回有效的置信区间（CI 下界 ≤ CI 上界），effect 列不含 NaN
3. **错误处理**: 无效输入（缺失列、越界参数、不支持的配置）均抛出清晰的 ValueError
4. **代码质量**: 修复了 ~50 个 bug，涵盖严重崩溃、静默数据损坏、边界条件等多个类别
5. **测试覆盖**: 80 项测试，覆盖 15 种组合、边界用例、数学正确性、集成场景

**测试结果: 80/80 通过 ✅**
