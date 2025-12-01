"""
自适应采样策略模块
Module for adaptive sampling strategies.
"""
import numpy as np
import pandas as pd
from typing import Protocol, Tuple

# 使用我们集中的环境检查来确定依赖是否可用
from ..utils.environment import KRIGING_AVAILABLE

if KRIGING_AVAILABLE:
    # 路径已由 environment.py 设置，可以直接导入
    from myKriging import training as kriging_training, testing as kriging_testing
else:
    # 如果Kriging不可用，我们定义一个占位符，以便代码至少可以被导入和分析
    # 但任何试图使用它的操作都会在运行时失败。
    def kriging_training(*args, **kwargs):
        raise ImportError("Kriging 模块未安装，无法进行训练。")
    def kriging_testing(*args, **kwargs):
        raise ImportError("Kriging 模块未安装，无法进行预测。")


class KrigingModel(Protocol):
    """
    这是一个协议类 (Protocol)，用于定义代理模型应具备的接口。
    AdaptiveSampler 将依赖此接口，而不是任何具体的实现。
    这使得未来可以轻松替换不同的代理模型（如高斯过程、神经网络等）。
    """
    def fit(self, points: np.ndarray, values: np.ndarray):
        ...

    def predict(self, points_to_predict: np.ndarray) -> np.ndarray:
        ...


class GpuKrigingSurrogate:
    """
    GPU加速的克里金模型的具体实现，作为代理模型使用。
    此类遵循 KrigingModel 协议。
    (原名 GPUKriging)
    """
    def __init__(self, variogram_model='exponential', **kwargs):
        if not KRIGING_AVAILABLE:
            raise RuntimeError("Kriging 模块不可用，无法实例化 GpuKrigingSurrogate。")
        self.model = None
        self._is_fitted = False
        self.variogram_model = variogram_model
        self.kriging_params = kwargs
        print(f"INFO: (GpuKrigingSurrogate) Initialized with variogram model: {self.variogram_model}")

    def fit(self, points: np.ndarray, values: np.ndarray):
        """
        使用稀疏的点坐标和对应的残差值来训练克里金代理模型。
        """
        print(f"INFO: (GpuKrigingSurrogate) Fitting model with {len(points)} points...")
        df = pd.DataFrame({
            'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2], 'target': values
        })

        self.model = kriging_training(
            df=df,
            variogram_model=self.variogram_model,
            nlags=self.kriging_params.get('nlags', 8),
            enable_plotting=False,
            weight=False,
            uk=False,
            cpu_on=False
        )
        self._is_fitted = True
        print("INFO: (GpuKrigingSurrogate) ✅ Model fitted.")

    def predict(self, points_to_predict: np.ndarray) -> np.ndarray:
        """
        对新的点进行批量预测，利用GPU加速。
        """
        if not self._is_fitted:
            raise RuntimeError("Kriging model must be fitted before prediction.")
        
        print(f"INFO: (GpuKrigingSurrogate) Predicting values for {len(points_to_predict)} points...")
        df_pred = pd.DataFrame({
            'x': points_to_predict[:, 0], 'y': points_to_predict[:, 1], 'z': points_to_predict[:, 2],
            'target': np.zeros(points_to_predict.shape[0])
        })

        predictions, _ = kriging_testing(
            df=df_pred, model=self.model,
            block_size=self.kriging_params.get('block_size', 10000),
            cpu_on=False, style="gpu_b", multi_process=False,
            print_time=False, torch_ac=False, compute_precision=False
        )
        print(f"INFO: (GpuKrigingSurrogate) ✅ Prediction complete.")
        return predictions.flatten()


class AdaptiveSampler:
    """
    自适应采样器。
    使用一个代理模型（如Kriging）来引导在下一训练周期中应该在哪里添加新的配置点。
    """
    def __init__(self, domain_bounds: np.ndarray, total_candidates: int = 100000):
        self.bounds = domain_bounds
        # 预先在整个域内生成大量的候选点，后续从中筛选
        self.candidate_points = np.random.rand(total_candidates, 3) * \
            (domain_bounds[1] - domain_bounds[0]) + domain_bounds[0]
        print(f"INFO: (AdaptiveSampler) Initialized with {total_candidates} candidate points.")

    def generate_new_collocation_points(
        self,
        surrogate_model: KrigingModel,
        num_points_to_sample: int,
        exploration_ratio: float
    ) -> np.ndarray:
        """
        使用代理模型引导生成新的配置点。

        Args:
            surrogate_model: 一个遵循KrigingModel协议的训练好的代理模型实例。
            num_points_to_sample: 需要生成的总点数。
            exploration_ratio: 探索率，取值在[0, 1]之间，代表随机采样的比例。

        Returns:
            新的配置点集 (np.ndarray)。
        """
        print(f"INFO: (AdaptiveSampler) Generating new points with exploration ratio: {exploration_ratio:.1%}")

        # 1. 使用代理模型预测所有候选点的残差
        predicted_residuals = surrogate_model.predict(self.candidate_points)
        print(f"    - Surrogate model predicted residuals stats (on {len(self.candidate_points)} candidates):")
        print(f"      - Max={np.max(predicted_residuals):.4e}, Min={np.min(predicted_residuals):.4e}, "
              f"Mean={np.mean(predicted_residuals):.4e}, Std={np.std(predicted_residuals):.4e}")

        # 2. "Exploitation": 找到预测残差最大的点 ("hard-case mining")
        num_exploitation_points = int(num_points_to_sample * (1 - exploration_ratio))
        hard_case_indices = np.argsort(predicted_residuals)[-num_exploitation_points:]
        exploitation_points = self.candidate_points[hard_case_indices]

        # 3. "Exploration": 加入一部分随机点以避免陷入局部最优
        num_exploration_points = num_points_to_sample - num_exploitation_points
        if num_exploration_points > 0:
            # 确保我们不会选择已经用于 "exploitation" 的点
            remaining_indices = np.setdiff1d(np.arange(len(self.candidate_points)), hard_case_indices, assume_unique=True)
            random_indices = np.random.choice(remaining_indices, num_exploration_points, replace=False)
            exploration_points = self.candidate_points[random_indices]
            
            print(f"INFO: (AdaptiveSampler) Generated {num_exploitation_points} exploitation points and {num_exploration_points} exploration points.")
            return np.vstack([exploitation_points, exploration_points])
        else:
            print(f"INFO: (AdaptiveSampler) Generated {num_exploitation_points} exploitation points (no exploration).")
            return exploitation_points
