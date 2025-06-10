"""
GPU-Accelerated Block Kriging × PINN 耦合重建工具模块
GPU-Accelerated Block Kriging × PINN Coupling Reconstruction Tools

功能概述 (Functionality Overview):
- 通用工具 (Common Tools): 数据标准化、误差统计、可视化
- 方案1专用 (Mode 1 Specific): PINN → 残差Kriging → 加权融合
- 方案2专用 (Mode 2 Specific): Kriging ROI样本扩充 → PINN重训练

作者: AI Assistant
日期: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time
from dataclasses import dataclass
from pathlib import Path

# ==================== 从新版PINN项目迁移的数据处理工具 ====================

class RadiationDataProcessor:
    """
    Enhanced data processor for radiation field data
    Supports multiple input formats including {z: DataFrame[y, x]} from tool.py
    """
    
    def __init__(self, space_dims=None, world_bounds=None):
        """
        Initialize the data processor
        
        Args:
            space_dims: Physical dimensions [x, y, z] in meters
            world_bounds: Dict with 'min' and 'max' arrays, or None for auto-detection
        """
        self.space_dims = np.array(space_dims) if space_dims is not None else None
        self.world_bounds = world_bounds
        self.dose_data = None
        
    def load_from_dict(self, data_dict: Dict, space_dims=None, world_bounds=None):
        """
        Load radiation data from dictionary format {z: DataFrame[y, x]} or {z: numpy_array}
        Compatible with tool.py RadiationDataset format
        
        Args:
            data_dict: Dictionary where keys are z-coordinates and values are 2D data (DataFrame or numpy array)
            space_dims: Physical dimensions [x, y, z] in meters
            world_bounds: Physical bounds dict with 'min' and 'max' keys
            
        Returns:
            dict: Standardized dose_data format for PINN usage
        """
        print("Loading radiation data from dictionary format...")
        
        z_coords = sorted(data_dict.keys())
        first_layer = data_dict[z_coords[0]]
        
        if hasattr(first_layer, 'values'):
            first_array = first_layer.values
        else:
            first_array = np.array(first_layer)
        
        y_size, x_size = first_array.shape
        z_size = len(z_coords)
        
        dose_grid = np.zeros((x_size, y_size, z_size), dtype=np.float32)
        
        for z_idx, z_coord in enumerate(z_coords):
            layer_data = data_dict[z_coord]
            layer_array = layer_data.values if hasattr(layer_data, 'values') else np.array(layer_data)
            dose_grid[:, :, z_idx] = layer_array.T.astype(np.float32)
        
        if space_dims is not None:
            self.space_dims = np.array(space_dims, dtype=np.float32)
        elif self.space_dims is None:
            self.space_dims = np.array([20.0, 10.0, 10.0], dtype=np.float32)
        
        if world_bounds is not None:
            self.world_bounds = world_bounds
            world_min = np.array(world_bounds['min'], dtype=np.float32)
            world_max = np.array(world_bounds['max'], dtype=np.float32)
        elif self.world_bounds is not None:
            world_min = np.array(self.world_bounds['min'], dtype=np.float32)
            world_max = np.array(self.world_bounds['max'], dtype=np.float32)
        else:
            world_min = -self.space_dims / 2.0
            world_max = self.space_dims / 2.0
        
        grid_shape = np.array([x_size, y_size, z_size])
        voxel_size = (world_max - world_min) / grid_shape
        
        self.dose_data = {
            'dose_grid': dose_grid, 'world_min': world_min, 'world_max': world_max,
            'voxel_size': voxel_size, 'grid_shape': grid_shape, 'space_dims': self.space_dims,
            'z_coords': np.array(z_coords, dtype=np.float32), 'original_data_dict': data_dict
        }
        return self.dose_data

    def load_from_numpy(self, dose_array, space_dims, world_bounds=None):
        """
        Load radiation data from 3D numpy array
        """
        if dose_array.ndim != 3:
            raise ValueError(f"Expected 3D array, got {dose_array.ndim}D")
        
        self.space_dims = np.array(space_dims, dtype=np.float32)
        grid_shape = np.array(dose_array.shape)
        
        if world_bounds is not None:
            world_min = np.array(world_bounds['min'], dtype=np.float32)
            world_max = np.array(world_bounds['max'], dtype=np.float32)
        else:
            world_min = -self.space_dims / 2.0
            world_max = self.space_dims / 2.0
        
        voxel_size = (world_max - world_min) / grid_shape
        
        self.dose_data = {
            'dose_grid': dose_array.astype(np.float32), 'world_min': world_min, 'world_max': world_max,
            'voxel_size': voxel_size, 'grid_shape': grid_shape, 'space_dims': self.space_dims
        }
        return self.dose_data

    def get_dose_data(self):
        if self.dose_data is None:
            raise ValueError("No data loaded.")
        return self.dose_data

class DataLoader:
    @staticmethod
    def load_dose_from_dict(data_dict: Dict, space_dims=None, world_bounds=None):
        processor = RadiationDataProcessor()
        return processor.load_from_dict(data_dict, space_dims, world_bounds)

    @staticmethod
    def load_dose_from_numpy(dose_array, space_dims, world_bounds=None):
        processor = RadiationDataProcessor()
        return processor.load_from_numpy(dose_array, space_dims, world_bounds)

# ==================== 耦合项目原有工具和模块导入 ====================

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

# 添加Kriging和新的PINN模块路径
sys.path.insert(0, str(project_root / "Kriging"))
sys.path.insert(0, str(project_root.parent / "PINN_claude"))

try:
    # 导入Kriging模块
    from myKriging import training as kriging_training, testing as kriging_testing
    from myPyKriging3D import MyOrdinaryKriging3D
    KRIGING_AVAILABLE = True
    print("✅ Kriging模块导入成功")
except ImportError as e:
    KRIGING_AVAILABLE = False
    warnings.warn(f"Kriging模块导入失败: {e}")

try:
    # 导入新的PINN模块
    from pinn_core import PINNTrainer, setup_deepxde_backend
    # 立即设置DeepXDE后端
    setup_deepxde_backend()
    PINN_AVAILABLE = True
    print("✅ 新版PINN模块 (pinn_core) 导入成功")
    print("✅ DeepXDE后端已设置为PyTorch")
except ImportError as e:
    PINN_AVAILABLE = False
    warnings.warn(f"新版PINN模块 (pinn_core) 导入失败: {e}")

# ==================== 全局常量与配置 ====================
# Global Constants and Configuration

EPSILON = 1e-30  # 数值稳定性常数
DEFAULT_METRICS = ['MAE', 'RMSE', 'MAPE', 'R2']

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
    kriging_variogram_model: str = "linear"
    kriging_block_size: int = 10000
    kriging_enable_uncertainty: bool = True  # 注意：当前实现可能不完全支持
    
    # PINN配置 PINN settings (对齐PINN子项目配置)
    pinn_epochs: int = 10000  # 与PINN子项目对齐：10000轮训练
    pinn_learning_rate: float = 1e-3
    pinn_network_layers: List[int] = None
    pinn_use_lbfgs: bool = True  # 启用L-BFGS，与PINN子项目对齐
    pinn_loss_weights: List[float] = None  # loss权重，与PINN子项目对齐
    pinn_sampling_strategy: str = 'positive_only'  # 采样策略，与PINN子项目对齐
    pinn_include_source: bool = False # 是否在PINN模型中包含源项参数化
    
    # 耦合配置 Coupling settings
    fusion_weight: float = 0.5  # 方案1中的权重ω
    roi_detection_strategy: str = 'high_density'  # 方案2中的ROI检测策略
    sample_augment_factor: float = 2.0  # 方案2中的样本扩充倍数
    
    def __post_init__(self):
        if self.pinn_network_layers is None:
            # 与PINN子项目对齐：使用无源PINN的网络配置 [3, 32, 32, 32, 32, 1]
            self.pinn_network_layers = [3, 32, 32, 32, 32, 1]
        
        if self.pinn_loss_weights is None:
            # 与PINN子项目对齐：使用无源PINN的loss权重 [1, 100]
            self.pinn_loss_weights = [1, 100]

# ==================== 通用工具 (Common Tools) ====================

@dataclass
class FieldTensor:
    """
    标准化的场数据结构
    Standardized field data structure
    """
    coordinates: np.ndarray  # (N, 3) - xyz坐标
    values: np.ndarray      # (N,) - 场值
    uncertainties: Optional[np.ndarray] = None  # (N,) - 不确定度
    metadata: Optional[Dict[str, Any]] = None   # 元数据
    
    def __post_init__(self):
        """验证数据一致性 Validate data consistency"""
        if self.coordinates.shape[0] != self.values.shape[0]:
            raise ValueError("坐标和数值的数量不匹配")
        if self.coordinates.shape[1] != 3:
            raise ValueError("坐标必须是3维 (x, y, z)")
        if self.uncertainties is not None and self.uncertainties.shape[0] != self.values.shape[0]:
            raise ValueError("不确定度和数值的数量不匹配")

@dataclass 
class ProbeSet:
    """
    标准化的测点数据结构
    Standardized probe data structure
    """
    positions: np.ndarray   # (N, 3) - 测点坐标
    measurements: np.ndarray # (N,) - 测量值
    weights: Optional[np.ndarray] = None  # (N,) - 权重
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """验证数据一致性"""
        if self.positions.shape[0] != self.measurements.shape[0]:
            raise ValueError("测点位置和测量值的数量不匹配")
        if self.positions.shape[1] != 3:
            raise ValueError("测点位置必须是3维 (x, y, z)")
        if self.weights is not None and self.weights.shape[0] != self.measurements.shape[0]:
            raise ValueError("权重和测量值的数量不匹配")

class DataNormalizer:
    """
    数据归一化工具
    Data normalization utilities
    """
    
    @staticmethod
    def normalize_tensor_to_grid(field_tensor: FieldTensor, 
                               grid_shape: Tuple[int, int, int],
                               world_bounds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        将张量数据转换为网格格式
        Convert tensor data to grid format
        
        Args:
            field_tensor: 输入的场数据张量
            grid_shape: 目标网格形状 (nx, ny, nz)
            world_bounds: 世界坐标边界 {'min': [x,y,z], 'max': [x,y,z]}
            
        Returns:
            Dict包含 'grid', 'coordinates', 'bounds'
        """
        coordinates = field_tensor.coordinates
        values = field_tensor.values
        
        world_min = world_bounds['min']
        world_max = world_bounds['max']
        
        # 创建规则网格
        x_grid = np.linspace(world_min[0], world_max[0], grid_shape[0])
        y_grid = np.linspace(world_min[1], world_max[1], grid_shape[1])  
        z_grid = np.linspace(world_min[2], world_max[2], grid_shape[2])
        
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        grid_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # 使用最近邻插值将不规则数据映射到网格
        from scipy.spatial import cKDTree
        tree = cKDTree(coordinates)
        distances, indices = tree.query(grid_coords)
        
        grid_values = values[indices].reshape(grid_shape)
        
        return {
            'grid': grid_values,
            'coordinates': grid_coords.reshape((*grid_shape, 3)),
            'bounds': world_bounds,
            'interpolation_distances': distances.reshape(grid_shape)
        }
    
    @staticmethod
    def robust_normalize(data: np.ndarray, 
                        quantile_range: Tuple[float, float] = (0.01, 0.99)) -> Tuple[np.ndarray, Dict]:
        """
        鲁棒归一化 (基于分位数)
        Robust normalization based on quantiles
        """
        q_low, q_high = quantile_range
        low_val = np.quantile(data, q_low)
        high_val = np.quantile(data, q_high)
        
        normalized = np.clip((data - low_val) / (high_val - low_val + EPSILON), 0, 1)
        
        normalization_info = {
            'method': 'robust',
            'low_val': low_val,
            'high_val': high_val,
            'quantile_range': quantile_range
        }
        
        return normalized, normalization_info

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

class VisualizationTools:
    """
    可视化工具集
    Visualization utilities
    """
    
    @staticmethod
    def plot_comparison_2d_slice(true_field: np.ndarray,
                               pred_field: np.ndarray, 
                               slice_axis: int = 2,
                               slice_idx: Optional[int] = None,
                               uncertainty_field: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None,
                               title_prefix: str = "") -> plt.Figure:
        """
        绘制2D切片对比图
        Plot 2D slice comparison
        
        Args:
            true_field: 真实场 (nx, ny, nz)
            pred_field: 预测场 (nx, ny, nz)
            slice_axis: 切片轴 (0=x, 1=y, 2=z)
            slice_idx: 切片索引，None则使用中间切片
            uncertainty_field: 不确定度场（可选）
            save_path: 保存路径（可选）
            title_prefix: 标题前缀
            
        Returns:
            matplotlib Figure对象
        """
        if slice_idx is None:
            slice_idx = true_field.shape[slice_axis] // 2
        
        # 提取切片
        if slice_axis == 0:
            true_slice = true_field[slice_idx, :, :]
            pred_slice = pred_field[slice_idx, :, :]
            uncertainty_slice = uncertainty_field[slice_idx, :, :] if uncertainty_field is not None else None
        elif slice_axis == 1:
            true_slice = true_field[:, slice_idx, :]
            pred_slice = pred_field[:, slice_idx, :]
            uncertainty_slice = uncertainty_field[:, slice_idx, :] if uncertainty_field is not None else None
        else:  # slice_axis == 2
            true_slice = true_field[:, :, slice_idx]
            pred_slice = pred_field[:, :, slice_idx]
            uncertainty_slice = uncertainty_field[:, :, slice_idx] if uncertainty_field is not None else None
        
        # 创建子图布局
        n_plots = 3 if uncertainty_slice is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        # 绘制真实场
        im1 = axes[0].imshow(true_slice.T, origin='lower', aspect='auto', 
                           norm=LogNorm(vmin=max(true_slice.min(), EPSILON), vmax=true_slice.max()))
        axes[0].set_title(f'{title_prefix}真实场 (轴{slice_axis}, 切片{slice_idx})')
        axes[0].set_xlabel('X' if slice_axis != 0 else ('Y' if slice_axis == 2 else 'Z'))
        axes[0].set_ylabel('Y' if slice_axis != 1 else ('X' if slice_axis == 2 else 'Z'))
        plt.colorbar(im1, ax=axes[0])
        
        # 绘制预测场
        im2 = axes[1].imshow(pred_slice.T, origin='lower', aspect='auto',
                           norm=LogNorm(vmin=max(pred_slice.min(), EPSILON), vmax=pred_slice.max()))
        axes[1].set_title(f'{title_prefix}预测场')
        axes[1].set_xlabel('X' if slice_axis != 0 else ('Y' if slice_axis == 2 else 'Z'))
        axes[1].set_ylabel('Y' if slice_axis != 1 else ('X' if slice_axis == 2 else 'Z'))
        plt.colorbar(im2, ax=axes[1])
        
        # 绘制不确定度场（如果提供）
        if uncertainty_slice is not None:
            im3 = axes[2].imshow(uncertainty_slice.T, origin='lower', aspect='auto')
            axes[2].set_title(f'{title_prefix}不确定度')
            axes[2].set_xlabel('X' if slice_axis != 0 else ('Y' if slice_axis == 2 else 'Z'))
            axes[2].set_ylabel('Y' if slice_axis != 1 else ('X' if slice_axis == 2 else 'Z'))
            plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    @staticmethod
    def plot_residual_analysis(residuals: np.ndarray,
                             coordinates: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        残差分析可视化
        Residual analysis visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 残差直方图
        axes[0, 0].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(residuals), color='red', linestyle='--', label=f'均值: {np.mean(residuals):.2e}')
        axes[0, 0].axvline(np.median(residuals), color='orange', linestyle='--', label=f'中位数: {np.median(residuals):.2e}')
        axes[0, 0].set_xlabel('残差值')
        axes[0, 0].set_ylabel('频率')
        axes[0, 0].set_title('残差分布直方图')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('残差Q-Q图 (正态性检验)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差绝对值vs预测值（如果有坐标信息）
        if coordinates is not None and coordinates.shape[0] == len(residuals):
            # 使用z坐标作为参考
            z_coords = coordinates[:, 2]
            scatter = axes[1, 0].scatter(z_coords, np.abs(residuals), alpha=0.6, c=np.abs(residuals), cmap='viridis')
            axes[1, 0].set_xlabel('Z坐标')
            axes[1, 0].set_ylabel('残差绝对值')
            axes[1, 0].set_title('残差绝对值 vs Z坐标')
            plt.colorbar(scatter, ax=axes[1, 0])
        else:
            # 残差绝对值vs索引
            axes[1, 0].plot(np.abs(residuals), 'o', alpha=0.6, markersize=3)
            axes[1, 0].set_xlabel('样本索引')
            axes[1, 0].set_ylabel('残差绝对值')
            axes[1, 0].set_title('残差绝对值分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 残差统计摘要
        axes[1, 1].axis('off')
        stats_text = f"""
残差统计摘要:

基本统计量:
• 样本数量: {len(residuals)}
• 均值: {np.mean(residuals):.4e}
• 标准差: {np.std(residuals):.4e}
• 最小值: {np.min(residuals):.4e}
• 最大值: {np.max(residuals):.4e}

分位数:
• 25%: {np.percentile(residuals, 25):.4e}
• 50%: {np.percentile(residuals, 50):.4e}
• 75%: {np.percentile(residuals, 75):.4e}

质量指标:
• MAE: {np.mean(np.abs(residuals)):.4e}
• RMSE: {np.sqrt(np.mean(residuals**2)):.4e}
• 偏度: {stats.skew(residuals):.4f}
• 峰度: {stats.kurtosis(residuals):.4f}
        """
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

    @staticmethod
    def plot_pinn_error_analysis(train_errors: np.ndarray, 
                                train_points: np.ndarray,
                                pinn_predictions: np.ndarray,
                                true_values: np.ndarray,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        PINN误差深度分析可视化
        PINN error deep analysis visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 误差vs真实值散点图
        axes[0, 0].scatter(true_values, train_errors, alpha=0.6, c='blue', s=20)
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测误差')
        axes[0, 0].set_title('误差 vs 真实值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(true_values, train_errors, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(true_values, p(true_values), "r--", alpha=0.8, linewidth=1)
        
        # 2. 误差vs预测值散点图
        axes[0, 1].scatter(pinn_predictions, train_errors, alpha=0.6, c='green', s=20)
        axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 1].set_xlabel('PINN预测值')
        axes[0, 1].set_ylabel('预测误差')
        axes[0, 1].set_title('误差 vs PINN预测值')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 3D空间误差分布
        ax_3d = fig.add_subplot(2, 3, 3, projection='3d')
        scatter = ax_3d.scatter(train_points[:, 0], train_points[:, 1], train_points[:, 2], 
                               c=np.abs(train_errors), cmap='hot', s=30, alpha=0.7)
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y') 
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('3D空间误差分布')
        plt.colorbar(scatter, ax=ax_3d, shrink=0.8)
        
        # 4. 误差累积分布函数
        sorted_errors = np.sort(np.abs(train_errors))
        y_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 0].plot(sorted_errors, y_vals, linewidth=2, color='purple')
        axes[1, 0].set_xlabel('误差绝对值')
        axes[1, 0].set_ylabel('累积概率')
        axes[1, 0].set_title('误差累积分布函数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加关键百分位线
        percentiles = [50, 80, 90, 95]
        colors = ['blue', 'orange', 'red', 'darkred']
        for p, color in zip(percentiles, colors):
            error_val = np.percentile(np.abs(train_errors), p)
            axes[1, 0].axvline(error_val, color=color, linestyle='--', alpha=0.7, 
                              label=f'{p}%: {error_val:.2e}')
        axes[1, 0].legend()
        
        # 5. 预测值vs真实值散点图
        axes[1, 1].scatter(true_values, pinn_predictions, alpha=0.6, c='cyan', s=20)
        
        # 完美预测线
        min_val = min(np.min(true_values), np.min(pinn_predictions))
        max_val = max(np.max(true_values), np.max(pinn_predictions))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
        
        axes[1, 1].set_xlabel('真实值')
        axes[1, 1].set_ylabel('PINN预测值')
        axes[1, 1].set_title('预测值 vs 真实值')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 误差统计摘要表格
        axes[1, 2].axis('off')
        
        # 计算相关性
        from scipy.stats import pearsonr, spearmanr
        pearson_corr, _ = pearsonr(true_values, pinn_predictions)
        spearman_corr, _ = spearmanr(true_values, pinn_predictions)
        
        stats_text = f"""
PINN预测性能详细分析:

基本误差统计:
• MAE: {np.mean(np.abs(train_errors)):.4e}
• RMSE: {np.sqrt(np.mean(train_errors**2)):.4e}
• MAPE: {np.mean(np.abs(train_errors)/(np.abs(true_values)+1e-8))*100:.2f}%
• 最大误差: {np.max(np.abs(train_errors)):.4e}

相关性分析:
• Pearson相关系数: {pearson_corr:.4f}
• Spearman相关系数: {spearman_corr:.4f}
• R²决定系数: {1 - np.sum(train_errors**2)/np.sum((true_values-np.mean(true_values))**2):.4f}

误差分布:
• 误差均值: {np.mean(train_errors):.4e}
• 误差标准差: {np.std(train_errors):.4e}
• 正偏误差比例: {np.sum(train_errors>0)/len(train_errors)*100:.1f}%
• 负偏误差比例: {np.sum(train_errors<0)/len(train_errors)*100:.1f}%

数据范围:
• 真实值范围: [{np.min(true_values):.2e}, {np.max(true_values):.2e}]
• 预测值范围: [{np.min(pinn_predictions):.2e}, {np.max(pinn_predictions):.2e}]
        """
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

# ==================== 通用接口适配器 ====================

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
    PINN模块的标准化接口适配器  
    Standardized interface adapter for PINN module
    """
    
    def __init__(self, config: ComposeConfig = None):
        self.config = config or ComposeConfig()
        self.trainer = None
        self.dose_data = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
           dose_data: Optional[Dict] = None,
           space_dims: List[float] = None,
           world_bounds: Dict = None,
           **kwargs) -> 'PINNAdapter':
        """
        根据输入数据和配置，训练或重新训练PINN模型。
        支持从dose_data自动初始化，或从space_dims/world_bounds手动初始化。
        """
        if not PINN_AVAILABLE:
            raise RuntimeError("新版PINN (pinn_core) 模块不可用，无法执行fit操作。")
            
        # 步骤 1: 用物理参数初始化 PINNTrainer
        self.trainer = PINNTrainer(physical_params=kwargs.get('physical_params'))

        # 步骤 2: 准备并调用 create_pinn_model
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        sampled_log_doses_values = np.log(y + EPSILON)
        network_config = kwargs.get('network_config')
        
        self.trainer.create_pinn_model(
            dose_data=dose_data,
            sampled_points_xyz=X,
            sampled_log_doses_values=sampled_log_doses_values,
            include_source=False,
            network_config=network_config
        )

        # 步骤 3: 准备并调用 train 方法
        train_params = {
            "epochs": self.config.pinn_epochs,
            "use_lbfgs": self.config.pinn_use_lbfgs,
            "loss_weights": self.config.pinn_loss_weights,
            "display_every": 500
        }
        train_params.update({k: v for k, v in kwargs.items() if k in ['epochs', 'use_lbfgs', 'loss_weights']})

        try:
            self.trainer.train(**train_params)
            self.trained = True
        except Exception as e:
            print(f"❌ PINN训练失败: {e}")
            raise e
            
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        标准化的predict接口
        
        Args:
            X: 预测点坐标 (N, 3)
            **kwargs: 额外的预测参数
            
        Returns:
            predictions: 预测值 (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用fit()")
            
        if not PINN_AVAILABLE:
            raise RuntimeError("PINN模块不可用")
        
        try:
            # 使用PINN进行预测
            log_predictions = self.trainer.predict(X)
            
            # 防止exp溢出的安全预测
            log_predictions = np.clip(log_predictions, -30, 15)  # 更合理的对数值范围
            predictions = np.exp(log_predictions) - EPSILON  # 转回原始尺度
            
            # 确保预测值为正数且在数据范围内
            # 动态确定合理的上界（基于训练数据范围）
            if hasattr(self, 'data_max_value'):
                max_pred = self.data_max_value * 10  # 允许一定的外推
            else:
                max_pred = 1e6  # 保守的上界
                
            predictions = np.clip(predictions, EPSILON, max_pred)
            
            if self.config.verbose:
                print(f"   🔍 PINN预测统计: 范围[{np.min(predictions):.2e}, {np.max(predictions):.2e}]")
                print(f"   📊 有效预测数量: {len(predictions)}")
            
            return predictions
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                if self.config.verbose:
                    print(f"❌ CUDA预测失败: {e}")
                    print("🔄 尝试CPU模式预测...")
                
                # 设置CPU模式重新预测
                import torch
                torch.set_default_device('cpu')
                
                log_predictions = self.trainer.predict(X)
                predictions = np.exp(log_predictions) - EPSILON
                
                return predictions
            else:
                raise e

def validate_compose_environment() -> Dict[str, bool]:
    """
    验证耦合环境的完整性
    Validate the coupling environment integrity
    
    Returns:
        Dict of availability status for each component
    """
    status = {
        'Kriging': KRIGING_AVAILABLE,
        'PINN': PINN_AVAILABLE, 
        'CuPy': CUPY_AVAILABLE,
        'PyTorch': TORCH_AVAILABLE
    }
    
    print("\n=== 环境检查结果 Environment Check ===")
    for component, available in status.items():
        status_str = "✅ 可用" if available else "❌ 不可用"
        print(f"{component}: {status_str}")
    
    return status 

# ==================== 方案1专用功能 (Mode 1 Specific) ====================
# PINN → 残差Kriging → 加权融合

class Mode1ResidualKriging:
    """
    方案1: 残差克里金插值专用工具
    Mode 1: Residual Kriging specific tools
    """
    
    def __init__(self, config: ComposeConfig = None):
        self.config = config or ComposeConfig()
        self.kriging_adapter = KrigingAdapter(config)
        
    def compute_residuals(self, 
                         train_points: np.ndarray,
                         train_values: np.ndarray, 
                         pinn_predictions: np.ndarray) -> np.ndarray:
        """
        计算PINN预测与真实值的残差
        Compute residuals between PINN predictions and true values
        
        Args:
            train_points: 训练点坐标 (N, 3)
            train_values: 真实训练值 (N,)
            pinn_predictions: PINN在训练点的预测值 (N,)
            
        Returns:
            residuals: 残差 = 真实值 - PINN预测值 (N,)
        """
        if len(train_values) != len(pinn_predictions):
            raise ValueError("真实值和PINN预测值的长度不匹配")
            
        residuals = train_values - pinn_predictions
        
        # 检查并修复异常值
        valid_mask = np.isfinite(residuals)
        if not np.all(valid_mask):
            print(f"       ⚠️ 发现 {np.sum(~valid_mask)} 个无效残差值，将进行修复")
            residuals = residuals[valid_mask]
            train_points = train_points[valid_mask]
            train_values = train_values[valid_mask]
            pinn_predictions = pinn_predictions[valid_mask]
        
        # 检查残差的数值特性
        residual_std = np.std(residuals)
        residual_range = np.max(residuals) - np.min(residuals)
        
        if residual_std < 1e-10 or residual_range < 1e-10:
            # 如果残差变化极小，说明PINN预测过于一致，需要添加空间结构
            print("       🔧 检测到残差空间变化过小，添加基于位置的微扰以改善Kriging建模")
            
            # 基于空间位置添加微扰，保持空间相关性
            spatial_weights = np.linalg.norm(train_points - np.mean(train_points, axis=0), axis=1)
            spatial_weights = (spatial_weights - np.min(spatial_weights)) / (np.max(spatial_weights) - np.min(spatial_weights) + 1e-10)
            
            # 添加与空间位置相关的微扰
            base_residual = np.mean(residuals)
            perturbation_scale = max(abs(base_residual) * 0.05, np.std(train_values) * 0.01, 1e-3)
            
            # 使用空间权重生成具有空间结构的微扰
            spatial_perturbation = perturbation_scale * (spatial_weights - 0.5) * 2
            random_perturbation = np.random.normal(0, perturbation_scale * 0.1, len(residuals))
            
            residuals = residuals + spatial_perturbation + random_perturbation
        
        # 对残差进行合理性检查和裁剪
        residuals = np.clip(residuals, -1e6, 1e6)
        
        if self.config.verbose:
            print(f"       📊 残差统计: 均值={np.mean(residuals):.4e}, 标准差={np.std(residuals):.4e}")
            print(f"       📈 残差范围: [{np.min(residuals):.4e}, {np.max(residuals):.4e}]")
            
        return residuals
        
    def residual_kriging(self,
                        train_points: np.ndarray,
                        train_values: np.ndarray,
                        pinn_predictions: np.ndarray,
                        prediction_points: np.ndarray,
                        return_uncertainty: bool = True,
                        **kriging_params) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        对残差进行克里金插值
        Perform Kriging interpolation on residuals
        
        Args:
            train_points: 训练点坐标 (N, 3)
            train_values: 真实训练值 (N,)
            pinn_predictions: PINN在训练点的预测值 (N,)
            prediction_points: 预测点坐标 (M, 3)
            return_uncertainty: 是否返回不确定度
            **kriging_params: 克里金参数
            
        Returns:
            residual_predictions: 残差预测 (M,)
            如果return_uncertainty=True: (residual_predictions, residual_std)
        """
        # 计算残差
        print(f"       🧮 计算残差 = 真实值 - PINN预测值...")
        residuals = self.compute_residuals(train_points, train_values, pinn_predictions)
        
        # 训练残差克里金模型
        print(f"       🏗️ 训练残差克里金模型 (变异函数: {kriging_params.get('variogram_model', 'linear')})...")
        self.kriging_adapter.fit(train_points, residuals, **kriging_params)
        print(f"       ✅ 残差克里金模型训练完成")
        
        # 预测残差
        if return_uncertainty and self.config.kriging_enable_uncertainty:
            residual_pred, residual_std = self.kriging_adapter.predict(
                prediction_points, return_std=True
            )
            return residual_pred, residual_std
        else:
            residual_pred = self.kriging_adapter.predict(prediction_points, return_std=False)
            if return_uncertainty:
                # 如果请求不确定度但不可用，返回零不确定度
                return residual_pred, np.zeros_like(residual_pred)
            else:
                return residual_pred

class Mode1Fusion:
    """
    方案1: 加权融合工具
    Mode 1: Weighted fusion tools
    """
    
    @staticmethod
    def fuse_residual(pinn_pred: np.ndarray,
                     kriging_residual: np.ndarray, 
                     weight: float = 0.5,
                     uncertainty: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        加权融合PINN预测和残差预测
        Weighted fusion of PINN predictions and residual predictions
        
        Args:
            pinn_pred: PINN预测值 (N,)
            kriging_residual: Kriging残差预测值 (N,)  
            weight: 残差权重 ω ∈ (0,1), 最终预测 = PINN + ω×残差
            uncertainty: Kriging残差的不确定度 (N,) [可选]
            
        Returns:
            fused_prediction: 融合预测 (N,)
            如果提供uncertainty: (fused_prediction, confidence_bounds)
        """
        if len(pinn_pred) != len(kriging_residual):
            raise ValueError("PINN预测和残差预测的长度不匹配")
            
        if not 0 < weight < 1:
            warnings.warn(f"权重 {weight} 不在推荐范围 (0,1) 内")
        
        # 加权融合
        fused_pred = pinn_pred + weight * kriging_residual
        
        if uncertainty is not None:
            # 计算置信界 (假设PINN无不确定度，只考虑Kriging残差的不确定度)
            # 95%置信界 ≈ ±1.96σ  
            confidence_bounds = weight * 1.96 * uncertainty
            return fused_pred, confidence_bounds
        else:
            return fused_pred
    
    @staticmethod
    def adaptive_weight_strategy(residuals: np.ndarray,
                               kriging_std: Optional[np.ndarray] = None,
                               strategy: str = 'variance_based') -> np.ndarray:
        """
        自适应权重策略
        Adaptive weighting strategy
        
        Args:
            residuals: 残差值 (N,)
            kriging_std: Kriging标准差 (N,) [可选]
            strategy: 权重策略 ('variance_based', 'magnitude_based', 'uniform')
            
        Returns:
            weights: 自适应权重 (N,)
        """
        n_points = len(residuals)
        
        if strategy == 'uniform':
            return np.full(n_points, 0.5)
        
        elif strategy == 'magnitude_based':
            # 基于残差幅度：残差越大，权重越高
            abs_residuals = np.abs(residuals)
            max_residual = np.max(abs_residuals) 
            weights = 0.1 + 0.8 * (abs_residuals / (max_residual + EPSILON))
            return np.clip(weights, 0.1, 0.9)
        
        elif strategy == 'variance_based' and kriging_std is not None:
            # 基于Kriging不确定度：不确定度越小，权重越高
            normalized_std = kriging_std / (np.max(kriging_std) + EPSILON)
            weights = 0.1 + 0.8 * (1 - normalized_std)  # 反比关系
            return np.clip(weights, 0.1, 0.9)
        
        else:
            warnings.warn(f"不支持的权重策略 '{strategy}' 或缺少必要数据，使用均匀权重")
            return np.full(n_points, 0.5)

# ==================== 方案2专用功能 (Mode 2 Specific) ====================  
# Kriging在ROI生成新样本 → 扩充数据 → 重新训练PINN

class Mode2ROIDetector:
    """
    方案2: 感兴趣区域(ROI)检测器
    Mode 2: Region of Interest (ROI) detector
    """
    
    @staticmethod
    def detect_roi(train_points: np.ndarray,
                  train_values: np.ndarray,
                  roi_strategy: str = 'high_density',
                  **strategy_params) -> Dict[str, np.ndarray]:
        """
        检测相关区域 (Region of Interest)
        Detect region of interest for sample augmentation
        
        Args:
            train_points: 训练点坐标 (N, 3)
            train_values: 训练点数值 (N,)
            roi_strategy: ROI检测策略
            **strategy_params: 策略相关参数
            
        Returns:
            roi_bounds: ROI边界信息 {'min': [x,y,z], 'max': [x,y,z], 'mask': bool_array}
        """
        if roi_strategy == 'high_density':
            return Mode2ROIDetector._detect_high_density_roi(
                train_points, train_values, **strategy_params
            )
        elif roi_strategy == 'high_value':
            return Mode2ROIDetector._detect_high_value_roi(
                train_points, train_values, **strategy_params
            )
        elif roi_strategy == 'bounding_box':
            return Mode2ROIDetector._detect_bounding_box_roi(
                train_points, train_values, **strategy_params
            )
        else:
            raise ValueError(f"不支持的ROI策略: {roi_strategy}")
    
    @staticmethod
    def _detect_high_density_roi(train_points: np.ndarray,
                               train_values: np.ndarray,
                               density_percentile: float = 75,
                               expansion_factor: float = 1.2) -> Dict[str, np.ndarray]:
        """高密度区域检测策略"""
        from scipy.spatial import cKDTree
        
        # 计算每个点的局部密度
        tree = cKDTree(train_points)
        # 计算到第5近邻的距离作为密度的逆指标
        k = min(5, len(train_points) - 1)
        distances, _ = tree.query(train_points, k=k+1)  # k+1因为包含自身
        local_density = 1 / (np.mean(distances[:, 1:], axis=1) + EPSILON)  # 排除自身
        
        # 选择高密度点
        density_threshold = np.percentile(local_density, density_percentile)
        high_density_mask = local_density >= density_threshold
        
        if not np.any(high_density_mask):
            # 如果没有高密度点，使用所有点
            high_density_mask = np.ones(len(train_points), dtype=bool)
        
        roi_points = train_points[high_density_mask]
        
        # 计算ROI边界
        roi_min = np.min(roi_points, axis=0)
        roi_max = np.max(roi_points, axis=0)
        
        # 扩展边界
        roi_center = (roi_min + roi_max) / 2
        roi_size = (roi_max - roi_min) * expansion_factor
        roi_min = roi_center - roi_size / 2
        roi_max = roi_center + roi_size / 2
        
        return {
            'min': roi_min,
            'max': roi_max, 
            'mask': high_density_mask,
            'density_scores': local_density
        }
    
    @staticmethod
    def _detect_high_value_roi(train_points: np.ndarray,
                             train_values: np.ndarray,
                             value_percentile: float = 80,
                             expansion_factor: float = 1.5) -> Dict[str, np.ndarray]:
        """高数值区域检测策略"""
        # 选择高数值点
        value_threshold = np.percentile(train_values, value_percentile)
        high_value_mask = train_values >= value_threshold
        
        if not np.any(high_value_mask):
            # 如果没有高数值点，使用数值大于0的点
            high_value_mask = train_values > 0
            
        if not np.any(high_value_mask):
            # 如果仍然没有，使用所有点
            high_value_mask = np.ones(len(train_points), dtype=bool)
        
        roi_points = train_points[high_value_mask]
        
        # 计算ROI边界并扩展
        roi_min = np.min(roi_points, axis=0)
        roi_max = np.max(roi_points, axis=0)
        
        roi_center = (roi_min + roi_max) / 2
        roi_size = (roi_max - roi_min) * expansion_factor
        roi_min = roi_center - roi_size / 2
        roi_max = roi_center + roi_size / 2
        
        return {
            'min': roi_min,
            'max': roi_max,
            'mask': high_value_mask,
            'value_scores': train_values
        }
    
    @staticmethod
    def _detect_bounding_box_roi(train_points: np.ndarray,
                               train_values: np.ndarray,
                               expansion_factor: float = 1.1) -> Dict[str, np.ndarray]:
        """包围盒ROI检测策略"""
        # 使用所有训练点的包围盒
        roi_min = np.min(train_points, axis=0)
        roi_max = np.max(train_points, axis=0)
        
        # 轻微扩展
        roi_center = (roi_min + roi_max) / 2
        roi_size = (roi_max - roi_min) * expansion_factor
        roi_min = roi_center - roi_size / 2
        roi_max = roi_center + roi_size / 2
        
        # 所有点都在ROI内
        all_points_mask = np.ones(len(train_points), dtype=bool)
        
        return {
            'min': roi_min,
            'max': roi_max,
            'mask': all_points_mask,
            'bounding_box': True
        }

class Mode2SampleAugmentor:
    """
    方案2: 样本扩充器  
    Mode 2: Sample augmentor using Kriging
    """
    
    def __init__(self, config: ComposeConfig = None):
        self.config = config or ComposeConfig()
        self.kriging_adapter = KrigingAdapter(config)
        
    def augment_by_kriging(self,
                          train_points: np.ndarray,
                          train_values: np.ndarray,
                          roi_bounds: Dict[str, np.ndarray],
                          augment_factor: float = 2.0,
                          sampling_strategy: str = 'grid',
                          **kriging_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        在ROI内用Kriging生成新样本
        Generate new samples in ROI using Kriging
        
        Args:
            train_points: 原始训练点坐标 (N, 3)
            train_values: 原始训练值 (N,)
            roi_bounds: ROI边界信息
            augment_factor: 扩充倍数 (新样本数 = 原样本数 × (augment_factor - 1))
            sampling_strategy: 采样策略 ('grid', 'random', 'adaptive')
            **kriging_params: Kriging参数
            
        Returns:
            augmented_points: 扩充后的坐标 (N+M, 3)
            augmented_values: 扩充后的数值 (N+M,)
        """
        # 训练Kriging模型
        self.kriging_adapter.fit(train_points, train_values, **kriging_params)
        
        # 生成ROI内的新采样点
        n_original = len(train_points) 
        n_new = int(n_original * (augment_factor - 1.0))
        
        if n_new <= 0:
            warnings.warn("扩充倍数太小，没有生成新样本")
            return train_points, train_values
        
        # 根据策略生成新采样点
        new_points = self._generate_sampling_points(
            roi_bounds, n_new, sampling_strategy, train_points
        )
        
        # 使用Kriging预测新点的数值
        new_values = self.kriging_adapter.predict(new_points, return_std=False)
        
        # 合并原始和新生成的样本
        augmented_points = np.vstack([train_points, new_points])
        augmented_values = np.concatenate([train_values, new_values])
        
        if self.config.verbose:
            print(f"样本扩充完成: {n_original} → {len(augmented_points)} 个样本")
            print(f"新样本数值范围: [{np.min(new_values):.4e}, {np.max(new_values):.4e}]")
        
        return augmented_points, augmented_values
    
    def _generate_sampling_points(self,
                                roi_bounds: Dict[str, np.ndarray], 
                                n_points: int,
                                strategy: str,
                                existing_points: np.ndarray) -> np.ndarray:
        """在ROI内生成采样点"""
        roi_min = roi_bounds['min']
        roi_max = roi_bounds['max']
        
        if strategy == 'grid':
            return self._generate_grid_points(roi_min, roi_max, n_points)
        elif strategy == 'random':
            return self._generate_random_points(roi_min, roi_max, n_points)
        elif strategy == 'adaptive':
            return self._generate_adaptive_points(roi_min, roi_max, n_points, existing_points)
        else:
            raise ValueError(f"不支持的采样策略: {strategy}")
    
    def _generate_grid_points(self, roi_min: np.ndarray, roi_max: np.ndarray, n_points: int) -> np.ndarray:
        """生成规则网格点"""
        # 计算每个维度的点数（尽量接近立方体）
        points_per_dim = int(np.ceil(n_points ** (1/3)))
        
        x = np.linspace(roi_min[0], roi_max[0], points_per_dim)
        y = np.linspace(roi_min[1], roi_max[1], points_per_dim)
        z = np.linspace(roi_min[2], roi_max[2], points_per_dim)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # 如果生成的点数超过需要的数量，随机选择
        if len(grid_points) > n_points:
            indices = np.random.choice(len(grid_points), n_points, replace=False)
            grid_points = grid_points[indices]
        
        return grid_points
    
    def _generate_random_points(self, roi_min: np.ndarray, roi_max: np.ndarray, n_points: int) -> np.ndarray:
        """生成随机采样点"""
        random_points = np.random.rand(n_points, 3)
        random_points = roi_min + random_points * (roi_max - roi_min)
        return random_points
    
    def _generate_adaptive_points(self, roi_min: np.ndarray, roi_max: np.ndarray, 
                                n_points: int, existing_points: np.ndarray) -> np.ndarray:
        """生成自适应采样点（避开已有点密集区域）"""
        from scipy.spatial import cKDTree
        
        # 构建已有点的KD树
        tree = cKDTree(existing_points)
        
        # 生成候选点（比需要的多一些）
        n_candidates = n_points * 3
        candidate_points = self._generate_random_points(roi_min, roi_max, n_candidates)
        
        # 计算每个候选点到最近已有点的距离
        distances, _ = tree.query(candidate_points)
        
        # 选择距离较大的点（远离已有点）
        sorted_indices = np.argsort(distances)[::-1]  # 降序排列
        selected_indices = sorted_indices[:n_points]
        
        return candidate_points[selected_indices]

# ==================== 端到端耦合工作流 ====================
# End-to-end coupling workflows

class CouplingWorkflow:
    """
    耦合工作流管理器
    Coupling workflow manager
    """
    
    def __init__(self, config: ComposeConfig = None):
        self.config = config or ComposeConfig()
        self.mode1_tools = {
            'residual_kriging': Mode1ResidualKriging(config),
            'fusion': Mode1Fusion()
        }
        self.mode2_tools = {
            'roi_detector': Mode2ROIDetector(),
            'augmentor': Mode2SampleAugmentor(config)
        }
        self.pinn_adapter = PINNAdapter(config)
        
    def run_mode1_pipeline(self,
                          train_points: np.ndarray,
                          train_values: np.ndarray,
                          prediction_points: np.ndarray,
                          fusion_weight: Optional[float] = None,
                          dose_data: Optional[Dict] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        执行方案1完整流程: PINN → 残差Kriging → 加权融合
        Execute Mode 1 complete pipeline
        """
        if fusion_weight is None:
            fusion_weight = self.config.fusion_weight
            
        results = {}
        
        # 步骤1: 训练PINN
        print("🔥 步骤1: 训练PINN模型...")
        self.pinn_adapter.fit(
            train_points, train_values,
            space_dims=kwargs.get('space_dims'),
            world_bounds=kwargs.get('world_bounds'),
            dose_data=dose_data,
            epochs=kwargs.get('epochs'),
            max_training_points=kwargs.get('max_training_points'),
            loss_weights=self.config.pinn_loss_weights,
            use_lbfgs=self.config.pinn_use_lbfgs
        )
        
        # 步骤2: PINN预测
        print("🔮 步骤2: PINN全场预测...")
        pinn_train_pred = self.pinn_adapter.predict(train_points)
        pinn_field_pred = self.pinn_adapter.predict(prediction_points)
        
        # ==================== 新增：详细PINN误差统计 ====================
        print("\n📊 步骤2.1: PINN误差分析...")
        
        # 训练点误差统计
        train_errors = train_values - pinn_train_pred
        train_metrics = {
            '训练集MAE': np.mean(np.abs(train_errors)),
            '训练集RMSE': np.sqrt(np.mean(train_errors**2)),
            '训练集MAPE': np.mean(np.abs(train_errors) / (np.abs(train_values) + EPSILON)) * 100,
            '训练集最大误差': np.max(np.abs(train_errors)),
            '训练集R²': 1 - np.sum(train_errors**2) / np.sum((train_values - np.mean(train_values))**2)
        }
        
        print("   🎯 PINN训练集性能:")
        for metric, value in train_metrics.items():
            print(f"      {metric}: {value:.4f}")
        
        # 预测值统计信息  
        print("   🔍 PINN预测统计: 范围[{:.2e}, {:.2e}]".format(np.min(pinn_train_pred), np.max(pinn_train_pred)))
        print("   📊 有效预测数量: {}".format(len(pinn_train_pred)))
        
        # 预测点预测值统计
        print("   🔍 PINN预测统计: 范围[{:.2e}, {:.2e}]".format(np.min(pinn_field_pred), np.max(pinn_field_pred)))
        print("   📊 有效预测数量: {}".format(len(pinn_field_pred)))
        
        # 添加误差分布统计
        error_percentiles = [5, 25, 50, 75, 95]
        error_stats = np.percentile(np.abs(train_errors), error_percentiles)
        print("   📈 训练误差分布 (绝对值):")
        for p, val in zip(error_percentiles, error_stats):
            print(f"      {p}%分位数: {val:.4e}")
        
        # 检查异常值
        error_threshold = np.mean(np.abs(train_errors)) + 3 * np.std(np.abs(train_errors))
        outlier_count = np.sum(np.abs(train_errors) > error_threshold)
        outlier_percentage = outlier_count / len(train_errors) * 100
        print(f"   ⚠️ 异常误差点: {outlier_count}个 ({outlier_percentage:.1f}%)")
        
        # 空间误差分析（如果训练点较多）
        if len(train_points) >= 10:
            # 计算误差的空间相关性
            spatial_distances = np.linalg.norm(train_points[:, None] - train_points[None, :], axis=2)
            error_correlations = []
            
            # 选择几个距离范围来分析误差相关性
            distance_ranges = [0.5, 1.0, 2.0, 5.0]
            for dist_range in distance_ranges:
                close_pairs = (spatial_distances > 0) & (spatial_distances < dist_range)
                if np.sum(close_pairs) > 0:
                    error_pairs = train_errors[:, None] * train_errors[None, :]
                    mean_error_corr = np.mean(error_pairs[close_pairs])
                    error_correlations.append((dist_range, mean_error_corr))
            
            if error_correlations:
                print("   🗺️ 空间误差相关性:")
                for dist, corr in error_correlations:
                    print(f"      距离<{dist:.1f}m: 相关性={corr:.4e}")
        
        # 存储误差统计结果
        results['pinn_train_errors'] = train_errors
        results['pinn_train_metrics'] = train_metrics
        results['pinn_predictions'] = pinn_field_pred
        # ==================== PINN误差统计结束 ====================
        
        # 步骤3: 残差Kriging
        print("⚡ 步骤3: 残差Kriging插值...")
        print(f"   🔍 计算PINN训练点预测与真实值的残差...")
        print(f"   🌐 对残差进行Kriging空间插值...")
        print(f"   📊 训练点数量: {len(train_points)}")
        print(f"   📍 预测点数量: {len(prediction_points)}")
        
        residual_pred, residual_std = self.mode1_tools['residual_kriging'].residual_kriging(
            train_points, train_values, pinn_train_pred, prediction_points,
            return_uncertainty=True, **kwargs.get('kriging_params', {})
        )
        
        print(f"   ✅ 残差Kriging插值完成")
        print(f"   📈 残差预测范围: [{np.min(residual_pred):.4e}, {np.max(residual_pred):.4e}]")
        if residual_std is not None:
            print(f"   📊 残差不确定度范围: [{np.min(residual_std):.4e}, {np.max(residual_std):.4e}]")
        results['residual_predictions'] = residual_pred
        results['residual_std'] = residual_std
        
        # 步骤4: 加权融合
        print("🔗 步骤4: 加权融合...")
        if residual_std is not None and not np.all(residual_std == 0):
            fused_pred, confidence_bounds = self.mode1_tools['fusion'].fuse_residual(
                pinn_field_pred, residual_pred, fusion_weight, residual_std
            )
            results['confidence_bounds'] = confidence_bounds
        else:
            fused_pred = self.mode1_tools['fusion'].fuse_residual(
                pinn_field_pred, residual_pred, fusion_weight
            )
            results['confidence_bounds'] = None
            
        results['final_predictions'] = fused_pred
        results['fusion_weight'] = fusion_weight
        
        print("✅ 方案1流程完成!")
        return results
    
    def run_mode2_pipeline(self,
                          train_points: np.ndarray, 
                          train_values: np.ndarray,
                          prediction_points: np.ndarray,
                          roi_strategy: Optional[str] = None,
                          augment_factor: Optional[float] = None,
                          dose_data: Optional[Dict] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        执行方案2完整流程: Kriging ROI样本扩充 → PINN重训练
        Execute Mode 2 complete pipeline  
        """
        if roi_strategy is None:
            roi_strategy = self.config.roi_detection_strategy
        if augment_factor is None:
            augment_factor = self.config.sample_augment_factor
            
        results = {}
        
        # 步骤1: ROI检测
        print("🎯 步骤1: 检测感兴趣区域(ROI)...")
        roi_bounds = self.mode2_tools['roi_detector'].detect_roi(
            train_points, train_values, roi_strategy, **kwargs.get('roi_params', {})
        )
        results['roi_bounds'] = roi_bounds
        
        # 步骤2: Kriging样本扩充
        print("📈 步骤2: Kriging样本扩充...")
        augmented_points, augmented_values = self.mode2_tools['augmentor'].augment_by_kriging(
            train_points, train_values, roi_bounds, augment_factor,
            **kwargs.get('kriging_params', {})
        )
        results['augmented_points'] = augmented_points
        results['augmented_values'] = augmented_values
        
        # 步骤3: 用扩充数据重新训练PINN
        print("🔥 步骤3: 用扩充数据重新训练PINN...")
        enhanced_pinn = PINNAdapter(self.config)
        enhanced_pinn.fit(
            augmented_points, augmented_values,
            space_dims=kwargs.get('space_dims'),
            world_bounds=kwargs.get('world_bounds'),
            dose_data=dose_data,
            epochs=kwargs.get('epochs'),
            loss_weights=self.config.pinn_loss_weights,
            use_lbfgs=self.config.pinn_use_lbfgs
        )
        
        # 步骤4: 最终预测
        print("🔮 步骤4: 增强PINN全场预测...")
        final_pred = enhanced_pinn.predict(prediction_points)
        results['final_predictions'] = final_pred
        results['enhanced_pinn'] = enhanced_pinn
        
        print("✅ 方案2流程完成!")
        return results

def print_compose_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         GPU Block-Kriging × PINN 耦合重建工具模块            ║  
    ║        GPU-Accelerated Block Kriging × PINN Coupling        ║
    ║                                                              ║
    ║  🚀 方案1: PINN → 残差Kriging → 加权融合                     ║
    ║  🎯 方案2: Kriging ROI样本扩充 → PINN重训练                  ║  
    ║                                                              ║
    ║  💡 支持GPU加速 | 🔬 物理约束 | 📊 不确定度量化              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

if __name__ == "__main__":
    print_compose_banner()
    validate_compose_environment() 