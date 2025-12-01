"""
自动化工作流模块
Module for automated workflows.
"""
import time
import numpy as np
from typing import Dict, Any, Optional

# 从 scikit-learn 导入依赖
from sklearn.neighbors import NearestNeighbors

# 从我们重构后的模块中导入适配器和模型
from ..models.kriging_adapter import KrigingAdapter
from ..models.pinn_adapter import AdvancedPINNAdapter, LegacyPINNAdapter
from ..models.pinn import PINNModel

EPSILON = 1e-30

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
            print("\n--- 正在执行高级 PINN 工作流 ---")
            
            # 2. 初始化核心 PINNModel
            pinn_config = self.config.get('pinn', {})
            training_data = np.hstack([train_points, train_values.reshape(-1, 1)])
            
            # 如果没有提供测试数据，使用训练数据的一部分
            if test_data is None:
                from sklearn.model_selection import train_test_split
                _, test_data = train_test_split(training_data, test_size=0.2, random_state=42)
            
            pinn_model_instance = PINNModel(
                dose_data=dose_data,
                training_data=training_data,
                test_data=test_data,
                **pinn_config.get('model_params', {})
            )

            # 3. 使用高级适配器来控制模型
            pinn_adapter = AdvancedPINNAdapter(pinn_model_instance)
            
            # 4. 生成配置点并执行训练周期
            model_params = pinn_config.get('model_params', {})
            num_collocation = model_params.get('num_collocation_points', 1024)
            world_min = dose_data['world_min']
            world_max = dose_data['world_max']
            collocation_points = np.random.rand(num_collocation, 3) * (world_max - world_min) + world_min
            
            training_params = pinn_config.get('training_params', {})
            pinn_adapter.train_cycle(
                max_epochs=training_params.get('total_epochs', 5000),
                detect_every=training_params.get('detect_every', 500),
                checkpoint_path_prefix=self.config.get('system', {}).get('checkpoint_path', './models/pinn_checkpoint'),
                collocation_points=collocation_points
            )
            
            # 5. 获取预测结果
            predictions = pinn_adapter.predict(prediction_points)
            results['predictions'] = predictions
            results['adapter'] = pinn_adapter

        end_time = time.time()
        results['total_time'] = end_time - start_time
        print(f"\n工作流 '{method_to_use}' 执行完毕，耗时 {results['total_time']:.2f} 秒。")
        
        return results
