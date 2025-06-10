"""
Physics-Informed Neural Networks (PINN) for radiation dose simulation
主入口模块 - 导入所有工具类和函数
"""

# 从各个模块导入主要类和函数
from .monte_carlo import GPUMonteCarloSimulator
from .visualization import Visualizer
from .data_processing import RadiationDataProcessor, DataLoader
from .pinn_core import SimulationConfig, PINNTrainer, ResultAnalyzer, setup_deepxde_backend

# 保持向后兼容性，导出所有主要组件
__all__ = [
    'SimulationConfig',
    'GPUMonteCarloSimulator', 
    'RadiationDataProcessor',
    'DataLoader',
    'PINNTrainer',
    'ResultAnalyzer',
    'Visualizer',
    'setup_deepxde_backend'
]

# 设置默认的后端
setup_deepxde_backend()
