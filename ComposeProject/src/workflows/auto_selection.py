"""
自动化工作流模块
Module for automated workflows.
"""
import time
import numpy as np
from typing import Dict, Any, Optional

# 从 scikit-learn 导入依赖
from sklearn.neighbors import NearestNeighbors

# 从我们重构后的模块中导入适配器
from ..models.kriging_adapter import KrigingAdapter
from .adaptive_experiment import run_adaptive_experiment

EPSILON = 1e-30

def _compute_coverage_ratio(points: np.ndarray, dose_data: Dict) -> float:
    """
    计算采样点覆盖度：采样点包围盒体积 / 整个域体积。
    用于避免点云只集中在局部但距离均匀导致 CV 偏低的误判。
    """
    if len(points) == 0:
        return 0.0
    domain_min = dose_data.get("world_min")
    domain_max = dose_data.get("world_max")
    if domain_min is None or domain_max is None:
        return 0.0
    domain_span = np.maximum(domain_max - domain_min, EPSILON)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_span = np.maximum(bbox_max - bbox_min, EPSILON)
    bbox_vol = float(np.prod(bbox_span))
    domain_vol = float(np.prod(domain_span))
    return min(1.0, bbox_vol / domain_vol)

class AutoSelectionWorkflow:
    """
    自动化模型选择与执行的工作流 (原名 CouplingWorkflow)。
    此类负责根据数据特征自动选择并执行最合适的模型（Kriging或PINN）。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化工作流。

        Args:
            config (Dict): 一个包含所有必要配置的字典，
                           例如: {'pinn': {...}, 'kriging': {...}, 'selection': {...}}
        """
        self.config = config
        print("INFO: (AutoSelectionWorkflow) 已初始化。")

    def analyze_data_distribution(self, points: np.ndarray, dose_data: Dict) -> str:
        """
        分析数据点的空间分布，以决定最优的预测方法。
        使用最近邻距离的变异系数(CV)来判断局部均匀性。
        """
        print("\n--- 正在分析数据分布 ---")
        
        # 先检查覆盖度，避免点只集中在局部却距离均匀导致的误判
        coverage_threshold = self.config.get("selection", {}).get("coverage_ratio_threshold", 0.3)
        coverage = _compute_coverage_ratio(points, dose_data)
        print(f"  - 覆盖度: {coverage:.3f} (阈值: {coverage_threshold})")
        if coverage < coverage_threshold:
            print("  - 决策: 覆盖不足，推荐使用 PINN（Kriging 外推风险高）。")
            return "pinn"

        if len(points) < self.config.get('selection', {}).get('min_points_for_kriging', 100):
            print(f"  - 决策: 数据点不足 ({len(points)})。推荐使用 PINN。")
            return 'pinn'

        # 计算每个点到其最近邻的距离
        nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(points)
        distances, _ = nn.kneighbors(points)
        nearest_distances = distances[:, 1]
        
        mean_dist = np.mean(nearest_distances)
        std_dist = np.std(nearest_distances)
        cv = std_dist / mean_dist if mean_dist > EPSILON else float('inf')
        
        threshold = self.config.get('selection', {}).get('uniformity_cv_threshold', 0.6)
        
        print(f"  - 最近邻CV值: {cv:.4f} (阈值: {threshold})")
        
        if cv < threshold:
            decision = 'kriging'
            print(f"  - 决策: 数据分布足够均匀。推荐使用 Kriging。")
        else:
            decision = 'pinn'
            print(f"  - 决策: 数据存在聚集或稀疏。推荐使用 PINN。")
            
        return decision

    def run(self,
            train_points: np.ndarray,
            train_values: np.ndarray,
            prediction_points: np.ndarray,
            dose_data: Dict,
            test_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        执行自动选择工作流。
        """
        start_time = time.time()
        results = {}

        # 1. 分析数据分布并决定使用哪种方法
        method_to_use = self.analyze_data_distribution(train_points, dose_data)
        results['method_used'] = method_to_use

        if method_to_use == 'kriging':
            print("\n--- 正在执行 Kriging 工作流 ---")
            
            # 2. 初始化并训练 Kriging 模型
            kriging_adapter = KrigingAdapter(
                kriging_config=self.config.get('kriging', {}),
                use_gpu=self.config.get('system',{}).get('use_gpu', True)
            )
            kriging_adapter.fit(train_points, train_values)

            # 3. 获取预测结果
            predictions, uncertainty = kriging_adapter.predict(prediction_points, return_std=True)
            results['predictions'] = predictions
            results['uncertainty'] = uncertainty
            results['adapter'] = kriging_adapter

        elif method_to_use == 'pinn':
            print("\n--- 正在执行自适应 PINN 工作流 (auto 判定) ---")
            adaptive_results = run_adaptive_experiment(
                self.config,
                return_payload=True,
                return_predictions=True,
            )
            predictions = None
            if adaptive_results:
                predictions = adaptive_results.get("predictions")
                results["adapter"] = adaptive_results.get("pinn")
                results["train_points"] = adaptive_results.get("train_points")
                results["train_values"] = adaptive_results.get("train_values")
                results["events"] = adaptive_results.get("events")
                results["adaptive_summary"] = adaptive_results.get("adaptive_summary")
                results["baseline_summary"] = adaptive_results.get("baseline_summary")
                results["time_seconds"] = adaptive_results.get("time_seconds")
            results['predictions'] = predictions
            results['method_used'] = 'pinn_adaptive'

        end_time = time.time()
        results['total_time'] = end_time - start_time
        print(f"\n工作流 '{method_to_use}' 执行完毕，耗时 {results['total_time']:.2f} 秒。")
        
        return results
