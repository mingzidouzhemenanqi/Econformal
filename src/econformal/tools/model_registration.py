import os
import importlib
from typing import Type

# 用户友好的别名映射：将包含特殊字符的名称映射到实际文件名
_CONFORMAL_ALIASES = {
    'jk+': 'jk_plus',
}


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
    # 别名映射：允许用户使用含特殊字符的友好名称
    model_name = _CONFORMAL_ALIASES.get(model_name, model_name)

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
    根据模型名称自动注册并返回对应的Conformal类
    
    Args:
        model_name (str): 模型名称，对应conformal_methods目录下的Python文件名（不含.py后缀）
        
    Returns:
        Type: 对应的Conformal类
        
    Raises:
        ValueError: 当指定的模型文件不存在时抛出异常
    """
    # 获取conformal_methods目录路径
    # conformal_methods_dir = os.path.join(os.path.dirname(__file__).parent, 'econometrics_methods')
    conformal_methods_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'econometrics_methods')
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
    module_path = f"..econometrics_methods.{model_name}"
    module = importlib.import_module(module_path,package=__package__)
    
    # 返回模块中的Conformal类
    return getattr(module, 'Econometric')