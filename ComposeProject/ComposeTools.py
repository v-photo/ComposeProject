"""
GPU-Accelerated Block Kriging × PINN 耦合重建工具模块
GPU-Accelerated Block Kriging × PINN Coupling Reconstruction Tools

功能概述 (Functionality Overview):
- 通用工具 (Common Tools): 数据标准化、误差统计、可视化
- 方案1专用 (Mode 1 Specific): PINN → Kriging → 加权融合

作者: AI Assistant
日期: 2024
"""

import sys
import numpy as np
from matplotlib.colors import LogNorm
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
# ==================== 耦合项目原有工具和模块导入 ====================
from PINN.pinn_core import  PINNTrainer

# 尝试导入所需的第三方库
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch不可用，部分GPU加速功能将被禁用")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy不可用，GPU加速功能将被禁用")

# 导入现有模块 - 需要确保路径正确
current_dir = Path(__file__).parent
project_root = current_dir.parent

# 添加Kriging模块路径
sys.path.insert(0, str(project_root / "Kriging"))

try:
    # 导入Kriging模块
    from myKriging import training as kriging_training, testing as kriging_testing
    from myPyKriging3D import MyOrdinaryKriging3D
    KRIGING_AVAILABLE = True
    print("✅ Kriging模块导入成功")
except ImportError as e:
    KRIGING_AVAILABLE = False
    warnings.warn(f"Kriging模块导入失败: {e}")

# 添加PINN模块路径
sys.path.insert(0, str(project_root / "PINN"))
try:
    from PINN.pinn_core import SimulationConfig, PINNTrainer, ResultAnalyzer
    from PINN.data_processing import DataLoader
    from PINN.visualization import Visualizer # <--- 修改这里
    from PINN.tools import setup_deepxde_backend
    from PINN.dataAnalysis import get_data
    PINN_AVAILABLE = True
    print("✅ PINN模块导入成功")
except ImportError as e:
    PINN_AVAILABLE = False
    warnings.warn(f"PINN模块导入失败: {e}")

# ==================== 全局常量与配置 ====================
# Global Constants and Configuration

EPSILON = 1e-30  # 数值稳定性常数
DEFAULT_METRICS = ['MAE', 'RMSE', 'MAPE', 'R2', 'MRE']

@dataclass
class ComposeConfig:
    """
    耦合系统全局配置类
    Global configuration for the coupling system
    """
    # 通用配置 Common settings
    gpu_enabled: bool = True
    verbose: bool = True
    random_seed: int = 42
    
    # Kriging配置 Kriging settings
    kriging_variogram_model: str = "exponential"
    kriging_block_size: int = 10000
    kriging_enable_uncertainty: bool = True  # 注意：当前实现可能不完全支持
    
    # [新] 自动方法选择阈值
    uniformity_cv_threshold: float = 0.6  # 最近邻距离的变异系数(CV)阈值，低于此值选Kriging

# ==================== 通用工具 (Common Tools) ====================
class MetricsCalculator:
    """
    误差统计计算器
    Error metrics calculator
    """
    
    @staticmethod
    def compute_metrics(true_values: np.ndarray, 
                       pred_values: np.ndarray,
                       metrics: List[str] = None) -> Dict[str, float]:
        """
        计算预测误差指标
        Compute prediction error metrics
        
        Args:
            true_values: 真实值
            pred_values: 预测值  
            metrics: 要计算的指标列表
            
        Returns:
            Dict[metric_name, metric_value]
        """
        if metrics is None:
            metrics = DEFAULT_METRICS
            
        # 确保输入为numpy数组
        true_values = np.asarray(true_values).flatten()
        pred_values = np.asarray(pred_values).flatten()
        
        if len(true_values) != len(pred_values):
            raise ValueError("真实值和预测值的长度不匹配")
        
        results = {}
        
        # 计算残差
        residuals = pred_values - true_values
        # 平均相对误差 Mean Relative Error
        if 'MRE' in metrics:
            results['MRE'] = np.mean(np.abs(residuals / true_values))
        
        # 平均绝对误差 Mean Absolute Error
        if 'MAE' in metrics:
            results['MAE'] = np.mean(np.abs(residuals))
        
        # 均方根误差 Root Mean Square Error  
        if 'RMSE' in metrics:
            results['RMSE'] = np.sqrt(np.mean(residuals**2))
        
        # 平均绝对百分比误差 Mean Absolute Percentage Error
        # 只在非零真值处计算
        if 'MAPE' in metrics:
            nonzero_mask = np.abs(true_values) > EPSILON
            if np.any(nonzero_mask):
                mape_values = np.abs(residuals[nonzero_mask] / true_values[nonzero_mask])
                results['MAPE'] = np.mean(mape_values) * 100  # 转换为百分比
            else:
                results['MAPE'] = float('inf')
        
        # 决定系数 R-squared
        if 'R2' in metrics:
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((true_values - np.mean(true_values))**2)
            results['R2'] = 1 - (ss_res / (ss_tot + EPSILON))
        
        # 相关系数 Pearson correlation
        if 'CORR' in metrics:
            correlation_matrix = np.corrcoef(true_values, pred_values)
            results['CORR'] = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        
        return results
    
    @staticmethod
    def compute_relative_error_stats(true_values: np.ndarray, 
                                   pred_values: np.ndarray,
                                   percentiles: List[float] = None) -> Dict[str, float]:
        """
        计算相对误差的统计分布
        Compute relative error statistics
        """
        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90, 95, 99]
        
        # 只在非零真值处计算相对误差
        nonzero_mask = np.abs(true_values) > EPSILON
        if not np.any(nonzero_mask):
            return {f'P{p}': float('inf') for p in percentiles}
        
        relative_errors = np.abs((pred_values[nonzero_mask] - true_values[nonzero_mask]) 
                                / true_values[nonzero_mask]) * 100
        
        stats = {}
        for p in percentiles:
            stats[f'P{p}'] = np.percentile(relative_errors, p)
        
        stats['mean_rel_error'] = np.mean(relative_errors)
        stats['std_rel_error'] = np.std(relative_errors)
        
        return stats

    def fuse_predictions(self,
                         pinn_pred: np.ndarray,
                         kriging_pred: np.ndarray,
                         kriging_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        [策略一] 基于Kriging不确定性阈值，进行硬切换融合.
        Final = w * Kriging + (1 - w) * PINN
        w = 1 if kriging_variance < threshold, else w = 0
        
        Args:
            pinn_pred: PINN的预测值 (N,)
            kriging_pred: Kriging的预测值 (N,)
            kriging_std: Kriging预测的标准差 (N,)
            
        Returns:
            (fused_prediction, fusion_weights): 融合后的预测和所使用的融合权重
        """
        threshold = self.config.kriging_variance_threshold
        
        # 1. 计算Kriging方差 (使用绝对值，不再归一化)
        kriging_variance = kriging_std**2
        
        # 2. 根据阈值生成二元(0或1)权重
        # 权重 w(x)=1 代表我们完全信任Kriging, w(x)=0 代表完全信任PINN
        fusion_weights = (kriging_variance < threshold).astype(np.float32)
        
        # 3. 执行加权融合
        fused_pred = fusion_weights * kriging_pred + (1 - fusion_weights) * pinn_pred
        
        if self.config.verbose:
            kriging_trusted_count = np.sum(fusion_weights)
            total_count = len(fusion_weights)
            trust_ratio = kriging_trusted_count / total_count * 100
            print("       - 融合权重统计 (策略一: 硬切换):")
            print(f"         - Kriging方差阈值: {threshold:.4e}")
            print(f"         - 信任Kriging的点数: {int(kriging_trusted_count)} / {total_count} ({trust_ratio:.2f}%)")

        return fused_pred, fusion_weights

# ==================== 模型适配器 (Model Adapters) ====================
class KrigingAdapter:
    """
    Kriging模块的标准化接口适配器
    Standardized interface adapter for Kriging module
    """
    
    def __init__(self, config: ComposeConfig = None):
        self.config = config or ComposeConfig()
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'KrigingAdapter':
        """
        标准化的fit接口
        Standardized fit interface
        
        Args:
            X: 训练点坐标 (N, 3)
            y: 训练点数值 (N,)
            **kwargs: 额外的kriging参数
            
        Returns:
            self
        """
        if not KRIGING_AVAILABLE:
            raise RuntimeError("Kriging模块不可用")
            
        # 将numpy数组转换为DataFrame格式（兼容现有接口）
        df = pd.DataFrame({
            'x': X[:, 0],
            'y': X[:, 1], 
            'z': X[:, 2],
            'target': y
        })
        
        # 使用现有的training函数
        variogram_model = kwargs.get('variogram_model', self.config.kriging_variogram_model)
        self.model = kriging_training(
            df=df,
            variogram_model=variogram_model,
            nlags=kwargs.get('nlags', 8),
            enable_plotting=kwargs.get('enable_plotting', False),
            weight=kwargs.get('weight', False),
            uk=kwargs.get('uk', False),
            cpu_on=not self.config.gpu_enabled
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        标准化的predict接口
        Standardized predict interface
        
        Args:
            X: 预测点坐标 (N, 3)
            return_std: 是否返回标准差（不确定度）
            **kwargs: 额外的预测参数
            
        Returns:
            predictions 或 (predictions, std)
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用fit()")
            
        if not KRIGING_AVAILABLE:
            raise RuntimeError("Kriging模块不可用")
        
        # 将预测点转换为DataFrame格式
        # 注意：这里需要提供虚拟的target列，但不会被使用
        df_pred = pd.DataFrame({
            'x': X[:, 0],
            'y': X[:, 1],
            'z': X[:, 2], 
            'target': np.zeros(X.shape[0])  # 虚拟值
        })
        
        # 使用现有的testing函数进行预测
        predictions, _ = kriging_testing(
            df=df_pred,
            model=self.model,
            block_size=kwargs.get('block_size', self.config.kriging_block_size),
            cpu_on=not self.config.gpu_enabled,
            style=kwargs.get('style', "gpu_b"),
            multi_process=kwargs.get('multi_process', False),
            print_time=kwargs.get('print_time', False),
            torch_ac=kwargs.get('torch_ac', False),
            compute_precision=False  # 关闭精度计算避免混淆
        )
        
        if return_std and self.config.kriging_enable_uncertainty:
            # 注意：当前实现可能不完全支持全局σ²输出
            # TODO: 需要修改现有Kriging代码以正确返回不确定度
            warnings.warn("当前Kriging实现暂不完全支持全局σ²输出，返回的不确定度可能不准确")
            
            # 临时方案：使用execute方法直接获取方差
            try:
                _, variances = self.model.execute(
                    style='points',
                    xpoints=X[:, 0],
                    ypoints=X[:, 1], 
                    zpoints=X[:, 2],
                    block_size=kwargs.get('block_size', self.config.kriging_block_size),
                    cpu_on=not self.config.gpu_enabled
                )
                std_values = np.sqrt(np.maximum(variances, 0))  # 确保非负
                return predictions, std_values
            except Exception as e:
                warnings.warn(f"获取不确定度失败: {e}，返回零不确定度")
                return predictions, np.zeros_like(predictions)
        else:
            return predictions

class PINNAdapter:
    """
    PINN模型的标准化接口适配器
    Standardized interface adapter for the PINN model
    """
    
    def __init__(self, physical_params: Dict, config: ComposeConfig = None):
        """
        初始化PINN适配器
        """
        self.config = config or ComposeConfig()
        if not PINN_AVAILABLE:
            raise RuntimeError("PINN模块不可用，无法创建PINNAdapter")
        if not physical_params:
            raise ValueError("PINNAdapter需要一个包含物理参数的字典 'physical_params'")
        
        self.trainer = PINNTrainer(physical_params=physical_params)
        self.dose_data = None  # 用于存储加载的数据
        self.is_fitted = False

    def fit_from_memory(self,
                        train_points: np.ndarray,
                        train_values: np.ndarray,
                        dose_data: Dict, 
                        sample_weights: Optional[np.ndarray] = None,
                        **kwargs) -> 'PINNAdapter':
        """
        使用内存中的训练数据点训练PINN模型。
        会自动处理对数转换。此方法专为耦合工作流设计。
        
        Args:
            train_points: 训练点坐标 (N, 3)
            train_values: 训练点数值 (N,)
            dose_data: 剂量数据字典
            sample_weights: 样本权重 (N,)，可选
            **kwargs: 其他参数
            
        Returns:
            self
        """
        print("INFO: 开始执行 PINNAdapter.fit_from_memory()")
        
        # 步骤 1: 数据准备 (转换物理值为对数值)
        print(f"      - 步骤1: 转换 {len(train_values)} 个训练点的物理值为对数值...")
        sampled_log_doses = np.log(np.maximum(train_values, EPSILON))
        print("      - 对数转换完成。")

        # 步骤 2: 创建并训练模型
        print("      - 步骤2: 创建并训练PINN模型...")
        network_config = kwargs.get('network_config', {'layers': [3] + [32] * 4 + [1], 'activation': 'tanh'})
        include_source = kwargs.get('include_source', False)
        
        # 检查是否提供了样本权重
        if sample_weights is not None:
            print(f"      - 检测到样本权重，将用于训练 (权重范围: [{np.min(sample_weights):.4f}, {np.max(sample_weights):.4f}])")
            # 确保权重长度与样本数量匹配
            if len(sample_weights) != len(train_points):
                raise ValueError(f"样本权重长度 ({len(sample_weights)}) 与训练点数量 ({len(train_points)}) 不匹配")
            
            # 存储样本权重供后续使用（如果PINN模块支持）
            self._sample_weights = sample_weights
        else:
            print("      - 未提供样本权重，使用均匀权重")
            self._sample_weights = None
            
        self.trainer.create_pinn_model(
            dose_data=dose_data,
            sampled_points_xyz=train_points,
            sampled_log_doses_values=sampled_log_doses,
            include_source=include_source,
            network_config=network_config
        )
        
        epochs = kwargs.get('epochs', 10000)
        use_lbfgs = kwargs.get('use_lbfgs', True)
        loss_weights = kwargs.get('loss_weights', [1, 100])
        
        # 如果有样本权重，尝试通过其他方式应用
        if self._sample_weights is not None:
            print("      - 注意: 当前PINN模块不直接支持样本权重，将通过其他方式实现")
            # 这里可以添加适用于您PINN模块的权重实现方式
            # 例如：通过修改损失函数、重复数据点等
        
        self.trainer.train(
            epochs=epochs, 
            use_lbfgs=use_lbfgs, 
            loss_weights=loss_weights
        )
        
        self.is_fitted = True
        print("INFO: PINNAdapter.fit_from_memory() 完成")
        return self

    def predict(self, prediction_points: np.ndarray) -> np.ndarray:
        """
        使用训练好的PINN模型进行预测。
        根据约定，此方法直接返回最终的物理剂量值(线性尺度)。
        """
        if not self.is_fitted:
            raise RuntimeError("PINN模型尚未训练，请先调用fit()")
            
        # 根据约定，trainer.predict()返回的就是最终的物理剂量
        predicted_doses = self.trainer.predict(prediction_points)
        
        return predicted_doses.flatten()

def validate_compose_environment() -> Dict[str, bool]:
    """
    验证耦合项目的核心依赖是否可用
    Validate the core dependencies for the coupling project
    
    Returns:
        Dict of availability status for each component
    """
    status = {
        'Kriging': KRIGING_AVAILABLE,
        'CuPy': CUPY_AVAILABLE,
        'PyTorch': TORCH_AVAILABLE
    }
    
    print("\n=== 环境检查结果 Environment Check ===")
    for component, available in status.items():
        status_str = "✅ 可用" if available else "❌ 不可用"
        print(f"{component}: {status_str}")
    
    return status 

# ==================== 方案1专用功能 (Mode 1 Specific) ====================
class Mode1Fusion:
    """
    方案1: 基于焦点区域的硬切换融合
    Mode 1: Hard-switch fusion based on focus region
    """
    def __init__(self, config: ComposeConfig = None):
        self.config = config or ComposeConfig()
    
    def fuse_predictions(self,
                         pinn_pred: np.ndarray,
                         kriging_pred: np.ndarray,
                         prediction_points: np.ndarray,
                         focus_center: np.ndarray,
                         focus_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        [策略二] 基于预定义的焦点区域，进行硬切换融合.
        Final = w * Kriging + (1 - w) * PINN
        w = 1 if point is inside focus_sphere, else w = 0
        
        Args:
            pinn_pred: PINN的预测值 (N,)
            kriging_pred: Kriging的预测值 (N,)
            prediction_points: 预测点的坐标 (N, 3)
            focus_center: 焦点区域的中心 (3,)
            focus_radius: 焦点区域的半径
            
        Returns:
            (fused_prediction, fusion_weights): 融合后的预测和所使用的融合权重
        """
        if focus_center is None or focus_radius is None:
            warnings.warn("未提供焦点区域参数，无法执行融合，将完全使用PINN结果。")
            return pinn_pred, np.zeros_like(pinn_pred)

        # 1. 计算所有预测点到焦点中心的距离
        distances_to_center = np.linalg.norm(prediction_points - focus_center, axis=1)
        
        # 2. 根据距离和半径生成二元(0或1)权重
        # 权重 w(x)=1 代表我们完全信任Kriging, w(x)=0 代表完全信任PINN
        fusion_weights = (distances_to_center <= focus_radius).astype(np.float32)
        
        # 3. 执行加权融合
        fused_pred = fusion_weights * kriging_pred + (1 - fusion_weights) * pinn_pred
        
        if self.config.verbose:
            kriging_trusted_count = np.sum(fusion_weights)
            total_count = len(fusion_weights)
            trust_ratio = kriging_trusted_count / total_count * 100
            print("       - 融合权重统计 (策略二: 焦点区域硬切换):")
            print(f"         - 焦点中心: {focus_center}, 半径: {focus_radius}")
            print(f"         - 信任Kriging的点数(在焦点区域内): {int(kriging_trusted_count)} / {total_count} ({trust_ratio:.2f}%)")

        return fused_pred, fusion_weights

# ==================== 端到端耦合工作流 ====================
# End-to-end coupling workflows

class CouplingWorkflow:
    """
    耦合工作流主编排器
    Main orchestrator for coupling workflows
    """
    def __init__(self, physical_params: Dict, config: ComposeConfig = None):
        """
        初始化工作流
        
        Args:
            physical_params: 物理参数字典 (如rho, mu)
            config: 全局配置对象
        """
        self.physical_params = physical_params
        self.config = config or ComposeConfig()
        
        if self.config.verbose:
            print("="*20 + " 耦合工作流初始化 " + "="*20)
            print(f"  - 使用配置: {self.config}")
            print("="*20 + " 初始化完成 " + "="*20 + "\n")

    def analyze_data_distribution(self, points: np.ndarray, dose_data: Dict) -> str:
        """
        分析数据点的空间分布，以决定最优的预测方法。
        使用两阶段检查：
        1. 全局分布：比较数据包围盒体积与总空间体积的比例。
        2. 局部均匀性：若全局分布通过，则使用最近邻距离的变异系数(CV)。

        Args:
            points: 训练数据点坐标 (N, D)
            dose_data: 包含世界边界和尺寸信息的字典。

        Returns:
            'kriging' 如果数据分布均匀。
            'pinn' 如果数据分布不均或呈聚集状态。
        """
        print("\n--- 步骤 1/3: 分析数据空间分布 ---")

        # --- 全局分布检查 ---
        total_volume = np.prod(dose_data['space_dims'])
        data_min = np.min(points, axis=0)
        data_max = np.max(points, axis=0)
        data_volume = np.prod(data_max - data_min)
        volume_ratio = data_volume / total_volume if total_volume > 0 else 0

        print(f"   - 全局分布检查:")
        print(f"     - 数据包围盒体积: {data_volume:.2f} m^3")
        print(f"     - 总空间体积: {total_volume:.2f} m^3")
        print(f"     - 体积占比: {volume_ratio:.2%}")

        # 假设体积占比小于30%即为显著聚集
        if volume_ratio < 0.3:
            print("   - 结论: 数据点显著聚集在部分空间。推荐使用 PINN 进行全局泛化。")
            return 'pinn'
        
        print("   - 全局分布通过，开始进行局部均匀性检查...")

        # --- 局部均匀性检查 ---
        # 1. 检查数据点数量
        if len(points) < 100: # 点太少，Kriging的变异函数估计不可靠
            print(f"   - 数据点数量: {len(points)} (< 100)")
            print("   - 结论: 数据点过少，Kriging模型可能不稳定。推荐使用 PINN。")
            return 'pinn'

        # 2. 计算每个点到其最近邻的距离
        nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(points)
        distances, _ = nn.kneighbors(points)
        
        # distances[:, 0] 是到自身的距离(0), distances[:, 1] 是到最近邻的距离
        nearest_distances = distances[:, 1]
        
        # 3. 计算距离的统计量
        mean_dist = np.mean(nearest_distances)
        std_dist = np.std(nearest_distances)
        cv = std_dist / mean_dist if mean_dist > EPSILON else float('inf')
        
        threshold = self.config.uniformity_cv_threshold
        
        # 4. 打印分析报告并做出决策
        print(f"   - 局部均匀性检查:")
        print(f"     - 训练点数量: {len(points)}")
        print(f"     - 最近邻平均距离: {mean_dist:.4f}")
        print(f"     - 最近邻距离标准差: {std_dist:.4f}")
        print(f"     - 变异系数 (CV): {cv:.4f} (值越低越均匀)")
        print(f"     - 决策阈值 (CV): {threshold}")
        
        if cv < threshold:
            decision = 'kriging'
            print(f"   - 结论: 数据在全局分布合理且局部均匀 (CV < {threshold})。推荐使用 Kriging。")
        else:
            decision = 'pinn'
            print(f"   - 结论: 数据虽全局分布，但局部存在聚集或空洞 (CV >= {threshold})。推荐使用 PINN。")
            
        return decision

    def run_auto_selection_pipeline(self,
                          train_points: np.ndarray,
                          train_values: np.ndarray,
                          prediction_points: np.ndarray,
                          dose_data: Optional[Dict] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        执行自动选择工作流：
        1. 分析训练数据的空间分布均匀性。
        2. 若分布均匀，则使用Kriging进行全局预测。
        3. 若分布不均，则使用PINN进行全局预测。
        """
        start_time = time.time()
        results = {}

        # 步骤 1: 分析数据分布并决定使用哪种方法
        method_to_use = self.analyze_data_distribution(train_points, dose_data)
        results['method_used'] = method_to_use

        if method_to_use == 'kriging':
            # --- 执行 Kriging 工作流 ---
            print("\n" + "-"*20 + " 执行 Kriging 全局预测 " + "-"*20)
            
            # 步骤 2: 数据清洗与Kriging模型训练
            print("\n--- 步骤 2/3: 清洗数据并训练Kriging模型 ---")
            mean_val = np.mean(train_values)
            std_val = np.std(train_values)
            threshold = 2 * std_val
            valid_mask = np.abs(train_values - mean_val) < threshold
            kr_train_points = train_points[valid_mask]
            kr_train_values = train_values[valid_mask]
            
            print(f"   - 原始训练点数: {len(train_values)}")
            print(f"   - 剔除异常值阈值 (mean + 2*std): {threshold:.4e}")
            print(f"   - 清洗后用于Kriging的训练点数: {len(kr_train_values)}")

            kriging_adapter = KrigingAdapter(self.config)
            kriging_adapter.fit(kr_train_points, kr_train_values)
            print("   ✅ Kriging模型训练完成。")

            # 步骤 3: 获取Kriging的预测结果
            print("\n--- 步骤 3/3: 获取Kriging在全场的独立预测 ---")
            kriging_predictions, kriging_std = kriging_adapter.predict(
                prediction_points, return_std=True
            )
            results['final_predictions'] = kriging_predictions
            results['kriging_uncertainty_std'] = kriging_std
            print(f"   - 已生成 {len(kriging_predictions)} 个Kriging预测。")

        elif method_to_use == 'pinn':
            # --- 执行 PINN 工作流 ---
            print("\n" + "-"*20 + " 执行 PINN 全局预测 " + "-"*20)
            
            # 步骤 2: 训练PINN模型
            print("\n--- 步骤 2/3: 使用全部稀疏数据训练PINN模型 ---")
            pinn_adapter = PINNAdapter(self.physical_params, self.config)
            pinn_adapter.fit_from_memory(
                train_points=train_points, 
                train_values=train_values, 
                dose_data=dose_data, 
                **kwargs
            )
            print("   ✅ PINN模型训练完成。")

            # 步骤 3: 获取PINN的预测结果
            print("\n--- 步骤 3/3: 获取PINN在全场的独立预测 ---")
            pinn_predictions = pinn_adapter.predict(prediction_points)
            results['final_predictions'] = pinn_predictions
            print(f"   - 已生成 {len(pinn_predictions)} 个PINN预测。")

        end_time = time.time()
        results['total_time'] = end_time - start_time
        print(f"\n方法 '{method_to_use}' pipeline 执行完毕，总耗时: {results['total_time']:.2f} 秒。")
        print("-" * 60)
        
        return results

def print_compose_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         GPU Block-Kriging & PINN 自动选择重建模块            ║  
    ║      Auto-Selector for GPU-Accelerated Kriging & PINN        ║
    ║                                                              ║
    ║  🚀 策略: 据数据分布均匀性，自动择优 (Kriging / PINN)        ║
    ║                                                              ║
    ║  💡 支持GPU加速 | 🔬 物理约束 | 📊 空间统计决策              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

if __name__ == "__main__":
    print_compose_banner()
    validate_compose_environment() 