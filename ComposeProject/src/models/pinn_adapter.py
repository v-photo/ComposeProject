"""
PINN 模型适配器模块
Module for the PINN model adapters.
"""
import numpy as np
from typing import Dict, Any, Optional

# 从集中的位置导入依赖检查和模块
from ..utils.environment import PINN_AVAILABLE
from .pinn import PINNModel

# 如果原始的PINN库可用，则导入它的Trainer
# 这是为了兼容旧的PINNAdapter
if PINN_AVAILABLE:
    from pinn_core import PINNTrainer
else:
    # 定义一个占位符以避免导入错误
    class PINNTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("旧版 'pinn_core' 模块不可用。")

EPSILON = 1e-30  # 避免log(0)

class LegacyPINNAdapter:
    """
    对旧版 PINNTrainer 的标准化接口适配器。(原名 PINNAdapter)
    此类封装了对一个更简单的、非自适应的PINN训练器的调用。
    """
    def __init__(self, physical_params: Dict[str, float]):
        if not PINN_AVAILABLE:
            raise RuntimeError("PINN模块不可用，无法创建LegacyPINNAdapter")
        
        self.trainer = PINNTrainer(physical_params=physical_params)
        self.is_fitted = False

    def fit(self,
            train_points: np.ndarray,
            train_values: np.ndarray,
            dose_data: Dict,
            **kwargs) -> 'LegacyPINNAdapter':
        """
        使用内存中的数据点训练旧版PINN模型。
        """
        print("INFO: (LegacyPINNAdapter) 正在使用旧版 PINNTrainer 训练模型...")
        
        # 1. 数据准备 (转换为对数尺度)
        log_values = np.log(np.maximum(train_values, EPSILON))

        # 2. 创建并训练模型
        network_config = kwargs.get('network_config', {'layers': [3] + [32] * 4 + [1], 'activation': 'tanh'})
        
        self.trainer.create_pinn_model(
            dose_data=dose_data,
            sampled_points_xyz=train_points,
            sampled_log_doses_values=log_values,
            include_source=kwargs.get('include_source', False),
            network_config=network_config
        )
        
        self.trainer.train(
            epochs=kwargs.get('epochs', 10000),
            use_lbfgs=kwargs.get('use_lbfgs', True),
            loss_weights=kwargs.get('loss_weights', [1, 100])
        )
        
        self.is_fitted = True
        print("INFO: (LegacyPINNAdapter) ✅ 模型训练完毕。")
        return self

    def predict(self, prediction_points: np.ndarray) -> np.ndarray:
        """使用训练好的模型进行预测。"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用fit()")
        
        predicted_doses = self.trainer.predict(prediction_points)
        return predicted_doses.flatten()

class AdvancedPINNAdapter:
    """
    高级PINN适配器，直接使用我们重构后的核心 PINNModel 类。
    它暴露了自适应训练所需的更精细的控制接口。
    """
    def __init__(self, pinn_model: PINNModel):
        self.model = pinn_model
        print("INFO: (AdvancedPINNAdapter) 已初始化。")
        
    def train_cycle(self, *args, **kwargs) -> Dict[str, Any]:
        """执行一个训练周期，直接调用PINNModel的同名方法。"""
        print(f"INFO: (AdvancedPINNAdapter) 正在开始训练周期...")
        return self.model.run_training_cycle(*args, **kwargs)
    
    def predict(self, points: np.ndarray) -> np.ndarray:
        """预测，直接调用PINNModel的同名方法。"""
        return self.model.predict(points)
    
    def update_loss_ratio(self, new_loss_ratio: float):
        """动态更新损失权重比值。"""
        self.model.update_loss_ratio(new_loss_ratio)
    
    def inject_new_data(self, new_data_array: np.ndarray):
        """注入新的训练数据。"""
        self.model.inject_new_data(new_data_array)
    
    def compute_pde_residual(self, points: np.ndarray) -> np.ndarray:
        """计算PDE残差。"""
        return self.model.compute_pde_residual(points)
    
    @property
    def mre_history(self):
        return self.model.mre_history
        
    @property
    def epoch_history(self):
        return self.model.epoch_history
