import os
import importlib
from typing import Type

# 用户友好的别名映射：将包含特殊字符的名称映射到实际文件名
_CONFORMAL_ALIASES = {
    'jk+': 'jk_plus',
    'cv+': 'cv_plus',
}

# 反向映射：实际文件名 → 用户友好名称（供 list_* 函数使用）
_CONFORMAL_REVERSE_ALIASES = {v: k for k, v in _CONFORMAL_ALIASES.items()}


def list_conformal_models():
    """返回可用的共形推断模型名称列表（用户友好名称）。"""
    import os as _os
    conformal_methods_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)), 'conformal_methods'
    )
    models = [
        f[:-3] for f in _os.listdir(conformal_methods_dir)
        if f.endswith('.py') and f not in ('__init__.py', 'conformal_base.py')
    ]
    return [_CONFORMAL_REVERSE_ALIASES.get(m, m) for m in models]


def list_econ_models():
    """返回可用的计量经济学模型名称列表。"""
    import os as _os
    econ_methods_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)), 'econometrics_methods'
    )
    models = [
        f[:-3] for f in _os.listdir(econ_methods_dir)
        if f.endswith('.py') and f != '__init__.py'
    ]
    return models


def get_conformal_model(model_name: str) -> Type:
    """
    根据模型名称自动注册并返回对应的Conformal类

    Args:
        model_name (str): 模型名称，对应conformal_methods目录下的Python文件名（不含.py后缀），
                          支持别名: 'jk+' → 'jk_plus'

    Returns:
        Type: 对应的Conformal类

    Raises:
        ValueError: 当指定的模型文件不存在时抛出异常
    """
    # 别名映射：允许用户使用含特殊字符的友好名称（不区分大小写）
    model_name_lower = model_name.lower()
    model_name = _CONFORMAL_ALIASES.get(model_name_lower, model_name_lower)

    # 获取conformal_methods目录路径,conformal_methods目录在上一级目录中
    # 获取当前文件的父目录

    conformal_methods_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conformal_methods')
    
    # 构建模块文件路径
    module_file_path = os.path.join(conformal_methods_dir, f"{model_name}.py")
    
    # 检查文件是否存在，排除__init__.py文件
    if not os.path.exists(module_file_path) or model_name == '__init__':
        available_models = [
            f[:-3] for f in os.listdir(conformal_methods_dir) 
            if f.endswith('.py') and f != '__init__.py'
        ]
        raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {available_models}")
    
    # 动态导入模块
    module_path = f"..conformal_methods.{model_name}"
    module = importlib.import_module(module_path,package=__package__)
    
    # 返回模块中的Conformal类
    return getattr(module, 'Conformal')


def get_econ_model(model_name: str) -> Type:
    """
    根据模型名称自动注册并返回对应的 Econometric 类

    Args:
        model_name (str): 模型名称，对应 econometrics_methods 目录下的 Python 文件名（不含 .py 后缀）

    Returns:
        Type: 对应的 Econometric 类

    Raises:
        ValueError: 当指定的模型文件不存在时抛出异常
    """
    # 转换为小写（文件名均为小写，不区分用户输入大小写）
    model_name = model_name.lower()

    # 获取 econometrics_methods 目录路径
    # econometrics_methods 目录在当前文件的父级目录中
    econ_methods_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'econometrics_methods')
    # 构建模块文件路径
    module_file_path = os.path.join(econ_methods_dir, f"{model_name}.py")

    # 检查文件是否存在，排除__init__.py文件
    if not os.path.exists(module_file_path) or model_name == '__init__':
        available_models = [
            f[:-3] for f in os.listdir(econ_methods_dir)
            if f.endswith('.py') and f != '__init__.py'
        ]
        raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {available_models}")

    # 动态导入模块
    module_path = f"..econometrics_methods.{model_name}"
    module = importlib.import_module(module_path, package=__package__)

    # 返回模块中的 Econometric 类
    return getattr(module, 'Econometric')