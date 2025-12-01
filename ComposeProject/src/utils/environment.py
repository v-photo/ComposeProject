"""
环境与依赖检查模块
Environment and dependency validation module.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict

def _check_dependencies():
    """
    检查核心依赖的可用性，并返回状态。
    此函数旨在在模块首次导入时内部调用一次。
    它返回可用性状态字典，并可能打印警告信息。
    """
    dependencies = {
        'torch': False,
        'cupy': False,
        'kriging': False,
        'pinn': False,
    }

    # 动态检查 Torch
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        warnings.warn("PyTorch不可用，部分GPU加速功能将被禁用。")

    # 动态检查 CuPy
    try:
        import cupy
        dependencies['cupy'] = True
    except ImportError:
        warnings.warn("CuPy不可用，GPU加速功能将被禁用。")

    # 动态检查 Kriging 模块
    try:
        # 此文件的路径为 .../ComposeProject/src/utils/environment.py
        # 项目根目录 (PINNproject) 在向上四层的位置
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        kriging_path = project_root / "Kriging"
        
        if kriging_path.is_dir():
            sys.path.insert(0, str(kriging_path))
            # 验证模块是否可以真的被导入
            from myKriging import training
            from myPyKriging3D import MyOrdinaryKriging3D
            dependencies['kriging'] = True
            print("✅ Kriging模块导入成功")
        else:
             warnings.warn(f"Kriging目录未找到: {kriging_path}")

    except ImportError as e:
        warnings.warn(f"Kriging模块导入失败: {e}")
    
    # 动态检查 PINN 模块
    try:
        pinn_path = project_root / "PINN"
        if pinn_path.is_dir():
            sys.path.insert(0, str(pinn_path))
            from data_processing import DataLoader
            from dataAnalysis import get_data
            dependencies['pinn'] = True
            print("✅ PINN模块导入成功")
        else:
            warnings.warn(f"PINN目录未找到: {pinn_path}")
    except ImportError as e:
        warnings.warn(f"PINN模块导入失败: {e}")

    return dependencies

# 模块加载时执行一次检查，并将结果存储在全局字典中
# 这可以确保检查只运行一次
DEPENDENCY_STATUS = _check_dependencies()

# 提供布尔常量，方便其他模块直接导入使用
TORCH_AVAILABLE = DEPENDENCY_STATUS['torch']
CUPY_AVAILABLE = DEPENDENCY_STATUS['cupy']
KRIGING_AVAILABLE = DEPENDENCY_STATUS['kriging']
PINN_AVAILABLE = DEPENDENCY_STATUS['pinn']


def validate_compose_environment() -> Dict[str, bool]:
    """
    显示已检查的核心依赖的状态。
    这个函数直接使用已经计算好的依赖状态。
    
    Returns:
        一个包含各组件可用状态的字典
    """
    status = {
        'Kriging': KRIGING_AVAILABLE,
        'CuPy': CUPY_AVAILABLE,
        'PyTorch': TORCH_AVAILABLE,
        'PINN': PINN_AVAILABLE
    }
    
    print("\n=== 环境检查结果 ===")
    for component, available in status.items():
        status_str = "✅ 可用" if available else "❌ 不可用"
        print(f"{component}: {status_str}")
    
    return status
