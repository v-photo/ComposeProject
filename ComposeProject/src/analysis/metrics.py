"""
误差统计计算模块
Error metrics calculation module.
"""

import numpy as np
from typing import Dict, List

# 数值稳定性常数
EPSILON = 1e-30

# 默认计算的指标
DEFAULT_METRICS = ['MAE', 'RMSE', 'MAPE', 'R2', 'MRE']

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
        
        # 使用一个掩码来处理所有可能除以零的情况
        nonzero_mask = np.abs(true_values) > EPSILON

        # 平均相对误差 Mean Relative Error
        if 'MRE' in metrics:
            if np.any(nonzero_mask):
                mre_values = np.abs(residuals[nonzero_mask] / true_values[nonzero_mask])
                results['MRE'] = np.mean(mre_values)
            else:
                # 如果所有真值都为零, 只有在预测值也全为零时误差才为0
                results['MRE'] = 0.0 if not np.any(residuals) else float('inf')

        # 平均绝对误差 Mean Absolute Error
        if 'MAE' in metrics:
            results['MAE'] = np.mean(np.abs(residuals))

        # 均方根误差 Root Mean Square Error
        if 'RMSE' in metrics:
            results['RMSE'] = np.sqrt(np.mean(residuals**2))

        # 平均绝对百分比误差 Mean Absolute Percentage Error
        if 'MAPE' in metrics:
            if np.any(nonzero_mask):
                mape_values = np.abs(residuals[nonzero_mask] / true_values[nonzero_mask])
                results['MAPE'] = np.mean(mape_values) * 100  # 转换为百分比
            else:
                results['MAPE'] = 0.0 if not np.any(residuals) else float('inf')

        # 决定系数 R-squared
        if 'R2' in metrics:
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((true_values - np.mean(true_values))**2)
            if ss_tot < EPSILON:
                results['R2'] = 1.0 if ss_res < EPSILON else 0.0
            else:
                results['R2'] = 1 - (ss_res / ss_tot)

        # 相关系数 Pearson correlation
        if 'CORR' in metrics:
            # 只有当两个数组都有变化时才计算
            if np.std(true_values) > EPSILON and np.std(pred_values) > EPSILON:
                correlation_matrix = np.corrcoef(true_values, pred_values)
                results['CORR'] = correlation_matrix[0, 1]
            else:
                results['CORR'] = 1.0 if np.std(true_values) < EPSILON and np.std(pred_values) < EPSILON else 0.0
        
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

        true_values = np.asarray(true_values).flatten()
        pred_values = np.asarray(pred_values).flatten()

        # 避免除以零
        relative_errors = np.abs(pred_values - true_values) / (np.abs(true_values) + EPSILON)

        stats = {
            'mean': np.mean(relative_errors),
            'std': np.std(relative_errors),
            'max': np.max(relative_errors),
            'min': np.min(relative_errors),
        }

        # 计算百分位数
        for p in percentiles:
            stats[f'p{p}'] = np.percentile(relative_errors, p)

        return stats
