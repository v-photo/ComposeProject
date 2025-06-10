#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
耦合项目 Version 2 - 纯PINN调用工具集

本模块旨在提供一个纯粹的、无依赖的PINN功能调用接口。
所有的核心功能，包括数据处理和PINN训练，都直接从 `PINN` 子项目导入和调用。
该版本完全移除了Kriging以及其他耦合逻辑，专注于提供一个清晰、直接的PINN代理。

设计原则:
- **高内聚**: 所有与PINN相关的调用都封装在 `PINNAdapterV2` 中。
- **低耦合**: 本模块不实现任何PINN或数据处理逻辑，仅作为调用方存在。
- **配置驱动**: 通过一个简化的 `ComposeConfigV2` 来管理所有PINN超参数。

使用方式:
1. 实例化 `ComposeConfigV2` 和 `PINNAdapterV2`。
2. 调用 `adapter.fit()` 方法来训练模型。
3. 调用 `adapter.predict()` 方法来进行预测。

作者: AI Assistant
日期: 2024-06-07
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# --- 路径设置 ---
# 确保可以从 PINN 子项目导入模块
try:
    # 定位到 /ComposeProject 目录
    current_dir = Path(__file__).parent.resolve()
    # 定位到项目根目录
    project_root = current_dir.parent
except NameError:
    # 在交互式环境中的回退方案
    project_root = Path('.').resolve()

# 将根目录和PINN目录添加到Python路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / 'PINN') not in sys.path:
    sys.path.insert(0, str(project_root / 'PINN'))

# --- 核心模块导入 ---
# 从PINN子项目直接导入所有需要的组件
try:
    from tools import PINNTrainer, RadiationDataProcessor, setup_deepxde_backend
    print("✅ (V2) 成功从 'PINN' 子项目导入核心模块 (PINNTrainer, RadiationDataProcessor)。")
    # 立即设置DeepXDE后端，确保环境一致性
    setup_deepxde_backend()
    print("✅ (V2) DeepXDE后端已成功设置。")
except ImportError as e:
    print(f"❌ (V2) 从 'PINN' 子项目导入模块失败。请确保路径正确且 `PINN/tools.py` 文件存在。")
    print(f"错误详情: {e}")
    sys.exit(1)


# --- 配置类 ---
@dataclass
class ComposeConfigV2:
    """
    一个简化的配置类，仅包含V2版本所需的PINN相关参数。
    """
    # PINN 训练配置
    epochs: int = 1000
    use_lbfgs: bool = False
    loss_weights: List[float] = field(default_factory=lambda: [1, 100])

    # PINN 网络结构配置
    network_layers: List[int] = field(default_factory=lambda: [3, 32, 32, 32, 32, 1])
    network_activation: str = 'tanh'
    network_initializer: str = 'Glorot uniform'

    def get_network_config(self) -> Dict[str, Any]:
        """返回DeepXDE兼容的网络配置字典"""
        return {
            'layers': self.network_layers,
            'activation': self.network_activation,
            'initializer': self.network_initializer
        }


# --- V2适配器 ---
class PINNAdapterV2:
    """
    第二版PINN适配器。
    
    这是一个围绕 `PINN.tools.PINNTrainer` 的轻量级封装。
    它不包含任何自己的逻辑，而是将所有任务委托给 `PINNTrainer`。
    """

    def __init__(self, config: Optional[ComposeConfigV2] = None):
        """
        初始化适配器。
        
        Args:
            config (ComposeConfigV2, optional): 配置对象。如果未提供，则使用默认配置。
        """
        self.config = config if config is not None else ComposeConfigV2()
        self.trainer: Optional[PINNTrainer] = None
        self.is_trained: bool = False
        print("✅ (V2) PINNAdapterV2 已初始化。")

    def fit(self, X: 'np.ndarray', y: 'np.ndarray', dose_data: Dict[str, Any], **kwargs) -> 'PINNAdapterV2':
        """
        训练PINN模型。
        
        此方法会创建一个 `PINNTrainer` 实例，并调用其 `create_pinn_model` 和 `train` 方法。
        所有传入的关键字参数将覆盖初始配置。

        Args:
            X (np.ndarray): 训练点的坐标，形状为 (N, 3)。
            y (np.ndarray): 训练点的剂量值，形状为 (N, 1) 或 (N,)。
            dose_data (Dict[str, Any]): 从 `RadiationDataProcessor` 获取的完整数据字典。
            **kwargs: 其他训练参数，可覆盖 `self.config` 中的设置。
                      例如: epochs, use_lbfgs, loss_weights, network_config。

        Returns:
            PINNAdapterV2: 返回自身以支持链式调用。
        """
        print("⏳ (V2) 开始PINN模型训练流程...")
        
        # 更新配置
        training_config = self._prepare_config(kwargs)

        # 1. 初始化 PINNTrainer
        # PINNTrainer本身是无状态的，这里可以传入物理参数，但对于无源项情况可以为空
        self.trainer = PINNTrainer()
        print("  - (V2) PINNTrainer 实例已创建。")

        # 2. 创建PINN模型
        # `create_pinn_model` 需要训练样本、网络配置等
        # 将 y 从 (N,) 调整为 (N, 1) 以符合DeepXDE要求
        y_train = y.reshape(-1, 1)
        
        print(f"  - (V2) 准备创建PINN模型，网络配置: {training_config['network_config']}")
        self.trainer.create_pinn_model(
            dose_data=dose_data,
            sampled_points_xyz=X,
            sampled_log_doses_values=y_train, # 注意：PINN内部可能使用log剂量
            network_config=training_config['network_config']
        )
        print("  - (V2) PINN模型已成功创建。")

        # 3. 训练模型
        print(f"  - (V2) 开始训练，共 {training_config['epochs']} 轮...")
        self.trainer.train(
            epochs=training_config['epochs'],
            use_lbfgs=training_config['use_lbfgs'],
            loss_weights=training_config['loss_weights']
        )
        self.is_trained = True
        print("✅ (V2) 模型训练完成。")
        
        return self

    def predict(self, X: 'np.ndarray') -> 'np.ndarray':
        """
        使用训练好的模型进行预测。

        Args:
            X (np.ndarray): 需要预测的点的坐标，形状为 (N, 3)。

        Returns:
            np.ndarray: 预测的剂量值，形状为 (N, 1)。
        """
        if not self.is_trained or self.trainer is None:
            raise RuntimeError("模型尚未训练，请先调用 `fit` 方法。")
        
        print(f"⏳ (V2) 正在对 {X.shape[0]} 个点进行预测...")
        predictions = self.trainer.predict(X)
        print("✅ (V2) 预测完成。")
        return predictions

    def get_learned_parameters(self) -> Optional[Dict[str, float]]:
        """
        如果模型包含可学习的物理参数（例如源项），则返回这些参数。
        """
        if not self.is_trained or self.trainer is None:
            print("⚠️ (V2) 模型尚未训练，无法获取学习参数。")
            return None
        
        return self.trainer.get_learned_parameters()

    def _prepare_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并默认配置和运行时传入的参数。
        """
        # 以默认配置为基础
        config_dict = {
            'epochs': self.config.epochs,
            'use_lbfgs': self.config.use_lbfgs,
            'loss_weights': self.config.loss_weights,
            'network_config': self.config.get_network_config()
        }
        
        # 允许 `network_config` 作为一个整体被覆盖
        if 'network_config' in kwargs:
            config_dict['network_config'] = kwargs['network_config']
        
        # 允许单个网络参数被覆盖
        if 'network_layers' in kwargs:
            config_dict['network_config']['layers'] = kwargs['network_layers']
        
        # 覆盖其他训练参数
        for key in ['epochs', 'use_lbfgs', 'loss_weights']:
            if key in kwargs:
                config_dict[key] = kwargs[key]
                
        return config_dict

if __name__ == '__main__':
    print("="*80)
    print("诊断信息: ComposeTools2.py 模块自检")
    print(f"项目根目录: {project_root}")
    print("Python 路径:")
    for p in sys.path:
        print(f"  - {p}")
    print("="*80)

    # 简单的实例化测试
    try:
        config = ComposeConfigV2()
        adapter = PINNAdapterV2(config)
        print("\n✅ (V2) 模块自检成功: ComposeConfigV2 和 PINNAdapterV2 实例化正常。")
    except Exception as e:
        print(f"\n❌ (V2) 模块自检失败: {e}")
