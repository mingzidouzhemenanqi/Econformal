# Econformal

计量经济学模型共形推断工具包 | Conformal Inference Toolkit for Econometric Models

[![PyPI version](https://badge.fury.io/py/econformal.svg)](https://badge.fury.io/py/econformal)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 简介 | Introduction

Econformal 是一个将共形推断（Conformal Inference）与计量经济学模型相结合的 Python 工具包。它提供了不确定性量化功能，为计量经济学预测提供统计保证。

Econformal is a Python package that combines conformal inference with econometric models, providing uncertainty quantification with statistical guarantees for econometric predictions.

## ✨ 主要功能 | Features

- **共形预测方法**：实现 Split Conformal、Full Conformal 等方法
- **计量经济学模型支持**：支持 DID（双重差分）、Synthetic Control（合成控制）等经典计量模型
- **不确定性量化**：为预测结果提供有效的置信区间
- **易于使用**：简洁的 API 设计，快速上手
- **可扩展性强**：支持自定义模型和预测方法

## 🚀 安装 | Installation

### 从 PyPI 安装

```bash
pip install econformal
```

### 从源码安装

```bash
git clone https://github.com/FORRYWU/econformal.git
cd econformal
pip install .
```

### 开发环境安装

```bash
pip install -e ".[dev]"
```

## 📖 快速开始 | Quick Start

```python
from econformal import Econformal

# 初始化模型
model = Econformal()

# 加载数据
# your data loading code here

# 拟合并预测
# model.fit()
# predictions = model.predict()

# 获取共形预测区间
# conformal_interval = model.conformal_predict()
```

## 📚 文档 | Documentation

详细文档请参考：[Documentation](https://github.com/FORRYWU/econformal#readme)

### 核心模块

- **conformal_methods**: 共形推断方法
  - `SplitConformal`: Split Conformal 预测
  - `FullConformal`: Full Conformal 预测
  
- **econometrics_methods**: 计量经济学方法
  - `DID`: 双重差分模型
  - `SyntheticControl`: 合成控制模型
  
- **tools**: 工具函数
  - `check`: 数据检查
  - `generate_data`: 数据生成
  - `plot`: 可视化
  - `model_registration`: 模型注册

## 📝 示例 | Examples

更多示例请参考 `example.py` 文件。

## 🤝 贡献 | Contributing

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证 | License

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📧 联系方式 | Contact

- Author: Forry Wu
- Email: your.email@example.com

## 🙏 致谢 | Acknowledgments

感谢所有为这个项目做出贡献的人！

---

**注意**: 本项目仍在开发中，如有问题请通过 GitHub Issues 反馈。