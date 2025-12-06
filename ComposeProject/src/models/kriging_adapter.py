"""
Kriging 模型适配器模块
Module for the Kriging model adapter.
"""
import numpy as np
import pandas as pd
import warnings
from typing import Union, Tuple, Dict, Any
import time
import torch

from ..utils.environment import KRIGING_AVAILABLE

# 如果Kriging库不可用，定义占位符函数
if KRIGING_AVAILABLE:
    from Kriging.myKriging import training as kriging_training, testing as kriging_testing
else:
    def kriging_training(*args, **kwargs):
        raise ImportError("Kriging 模块未安装，无法进行训练。")
    def kriging_testing(*args, **kwargs):
        raise ImportError("Kriging 模块未安装，无法进行预测。")

class KrigingAdapter:
    """
    Kriging模块的标准化接口适配器。
    它将外部 myKriging 库的调用封装在一个标准的 fit/predict 接口中。
    """
    
    def __init__(self, kriging_config: Dict[str, Any], use_gpu: bool = True):
        """
        初始化Kriging适配器。
        
        Args:
            kriging_config (Dict): 包含Kriging模型参数的配置字典。
            use_gpu (bool): 是否尝试使用GPU。
        """
        if not KRIGING_AVAILABLE:
            raise RuntimeError("Kriging模块不可用，无法创建KrigingAdapter。")
        
        backend = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.kriging_config = kriging_config
        self.model = None
        self.is_fitted = False
        print(f"INFO: (KrigingAdapter) 已在 {backend} 后端上初始化。")

    def fit(self, train_points: np.ndarray, train_values: np.ndarray) -> 'KrigingAdapter':
        """
        使用数据点训练(拟合)Kriging模型。
        """
        print(f"INFO: (KrigingAdapter) 正在用 {train_points.shape[0]} 个数据点拟合模型...")
        start_time = time.time()
        train_points = np.asarray(train_points).reshape(-1, 3)
        train_values = np.asarray(train_values).reshape(-1)
        df = pd.DataFrame({
            'x': train_points[:, 0],
            'y': train_points[:, 1],
            'z': train_points[:, 2],
            'target': train_values
        })
        self.model = kriging_training(
            df=df,
            variogram_model=self.kriging_config.get('variogram_model', 'exponential'),
            nlags=self.kriging_config.get('nlags', 8),
            enable_plotting=self.kriging_config.get('enable_plotting', False),
            weight=self.kriging_config.get('weight', False),
            uk=self.kriging_config.get('uk', False),
            cpu_on=not self.use_gpu
        )
        
        end_time = time.time()
        print(f"INFO: (KrigingAdapter) ✅ 模型拟合完毕，耗时 {end_time - start_time:.2f} 秒。")
        self.is_fitted = True
        return self

    def predict(self, prediction_points: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        使用拟合好的模型进行预测。
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用fit()")
            
        print(f"INFO: (KrigingAdapter) 正在为 {prediction_points.shape[0]} 个点生成预测...")
        df_pred = pd.DataFrame({
            'x': prediction_points[:, 0],
            'y': prediction_points[:, 1],
            'z': prediction_points[:, 2],
            'target': np.zeros(prediction_points.shape[0])
        })
        predictions, _ = kriging_testing(
            df=df_pred,
            model=self.model,
            block_size=self.kriging_config.get('block_size', 10000),
            cpu_on=not self.use_gpu,
            style=self.kriging_config.get('style', "gpu_b"),
            multi_process=self.kriging_config.get('multi_process', False),
            print_time=self.kriging_config.get('print_time', False),
            torch_ac=self.kriging_config.get('torch_ac', False),
            compute_precision=False
        )
        predictions = predictions.flatten()
        if return_std:
            # myKriging.testing 未返回 std，这里返回零占位
            return predictions, np.zeros_like(predictions)
        return predictions
