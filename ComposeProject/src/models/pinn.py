"""
核心PINN模型定义模块
Module for the core PINN model definition.
"""
import os
import numpy as np
import deepxde as dde
from pathlib import Path
from typing import Dict, List, Any

# 从新位置导入已重构的模块
from ..training.callbacks import EarlyCycleStopper


class PINNModel:
    """
    物理信息神经网络（PINN）的核心实现。
    这个类被设计为可从外部控制的模式，以支持自适应训练流程。
    它封装了模型的创建、编译、训练周期管理、数据注入和预测等功能。
    """
    def __init__(self,
                 dose_data: Dict,
                 training_data: np.ndarray,
                 test_data: np.ndarray,
                 num_collocation_points: int,
                 network_layers: List[int],
                 learning_rate: float = 1e-3,
                 loss_ratio: float = 10.0):
        """
        初始化PINN模型，但不立即开始训练。

        Args:
            dose_data (dict): 从DataLoader加载的数据字典。
            training_data (np.ndarray): 稀疏训练数据 [x,y,z,value]。
            test_data (np.ndarray): 稀疏测试数据 [x,y,z,value]。
            num_collocation_points (int): 求解域点的数量。
            network_layers (list): 神经网络结构。
            learning_rate (float): 学习率。
            loss_ratio (float): 数据损失权重与物理损失权重的比值。
        """
        print("INFO: (PINNModel) Initializing a DeepXDE-based model for external control...")
        
        self.test_data_linear = test_data
        
        # 1. 定义几何和可训练参数
        world_min = dose_data['world_min']
        world_max = dose_data['world_max']
        self.geometry = dde.geometry.Cuboid(world_min, world_max)
        self.log_k_pinn = dde.Variable(np.log(1.0))
        
        # 2. 构建PDE函数
        self.pde = self._build_pde_func()
        
        # 3. 准备数据对象
        observe_x = training_data[:, :3]
        observe_y = np.log(np.maximum(training_data[:, 3:], 1e-30))
        data_points = dde.icbc.PointSetBC(observe_x, observe_y, component=0)
        
        self.data = dde.data.PDE(
            self.geometry, self.pde, [data_points],
            num_domain=num_collocation_points, anchors=observe_x,
        )
        
        # 4. 创建网络和模型
        self.net = dde.nn.FNN(network_layers, "tanh", "Glorot normal")
        self.model = dde.Model(self.data, self.net)
        
        self.lr = learning_rate
        self.loss_ratio = loss_ratio
        self.mre_history = []
        self.epoch_history = []
        
        # 5. 编译模型
        self.compile_model()
        print(f"INFO: (PINNModel) ✅ Model compiled with loss_ratio={loss_ratio:.1f} (data/physics weight ratio).")
        
    def compile_model(self):
        """将模型编译封装成一个方法，方便重用。"""
        loss_weights = [1.0, self.loss_ratio]
        self.model.compile(
            "adam", lr=self.lr, loss_weights=loss_weights,
            external_trainable_variables=[self.log_k_pinn],
            metrics=[self.mean_relative_error_metric]
        )

    def update_loss_ratio(self, new_loss_ratio: float):
        """动态更新损失权重比值并重新编译模型。"""
        if abs(self.loss_ratio - new_loss_ratio) > 1e-6:
            old_ratio = self.loss_ratio
            self.loss_ratio = new_loss_ratio
            self.compile_model()
            print(f"INFO: (PINNModel) 损失权重比值已更新: {old_ratio:.2f} → {new_loss_ratio:.2f}")
            self.model.train(iterations=0, display_every=100000)
        else:
            print(f"INFO: (PINNModel) 损失权重比值未变化，跳过重编译 (当前: {self.loss_ratio:.2f})")

    def mean_relative_error_metric(self, y_true_ignored, y_pred_ignored) -> float:
        """
        自定义的指标函数，使用自己存储的测试集进行评估。
        """
        test_x = self.test_data_linear[:, :3]
        pred_y_log = self.model.predict(test_x, operator=None) # 确保不应用PDE算子
        
        pred_y_linear = np.exp(pred_y_log)
        true_y_linear = self.test_data_linear[:, 3:]
        
        mre = np.mean(np.abs(true_y_linear - pred_y_linear) / (np.abs(true_y_linear) + 1e-10))
        
        current_epoch = self.model.train_state.step or 0
        if not self.epoch_history or self.epoch_history[-1] != current_epoch:
            self.mre_history.append(mre)
            self.epoch_history.append(current_epoch)
            
        return mre

    def inject_new_data(self, new_data_array: np.ndarray):
        """向模型中注入新的训练数据点。"""
        print(f"\nINFO: (PINNModel) Injecting {len(new_data_array)} new data points...")
        current_bc = self.data.bcs[0]
        current_points = current_bc.points
        current_values_log = current_bc.values.cpu()

        new_points = new_data_array[:, :3]
        new_values_log = np.log(np.maximum(new_data_array[:, 3:], 1e-30)).reshape(-1, 1)

        combined_points = np.vstack([current_points, new_points])
        combined_values_log = np.vstack([current_values_log, new_values_log])
        
        new_bc = dde.icbc.PointSetBC(combined_points, combined_values_log, component=0)
        new_data_obj = dde.data.PDE(
            self.geometry, self.pde, [new_bc],
            num_domain=self.data.num_domain, anchors=combined_points
        )
        
        self.data = new_data_obj
        self.model.data = self.data
        self.compile_model()
        print("INFO: (PINNModel) ✅ Model re-compiled with new data.")
        self.model.train(iterations=0, display_every=100000)

    def _build_pde_func(self):
        """将PDE定义封装在一个工厂函数中，以捕获self.log_k_pinn。"""
        def pde_func(x, u):
            grad_u_sq = dde.grad.jacobian(u, x, i=0, j=0)**2 + \
                        dde.grad.jacobian(u, x, i=0, j=1)**2 + \
                        dde.grad.jacobian(u, x, i=0, j=2)**2
            laplacian_u = dde.grad.hessian(u, x, i=0, j=0) + \
                          dde.grad.hessian(u, x, i=1, j=1) + \
                          dde.grad.hessian(u, x, i=2, j=2)
            k_squared = dde.backend.exp(2 * self.log_k_pinn)
            return grad_u_sq + laplacian_u - k_squared
        return pde_func

    def run_training_cycle(self,
                           max_epochs: int,
                           detect_every: int,
                           collocation_points: np.ndarray,
                           checkpoint_path_prefix: str,
                           detection_threshold: float = 0.1) -> Dict[str, Any]:
        """
        执行一个带有动态停止条件的训练周期。
        """
        os.makedirs(Path(checkpoint_path_prefix).parent, exist_ok=True)

        # 更新配置点
        if self.model.train_state.X_train is None:
            self.model.train(iterations=0)
        num_bc_points = self.data.bcs[0].points.shape[0]
        start_index = num_bc_points
        end_index = len(self.model.train_state.X_train) - len(self.data.anchors)
        self.model.train_state.X_train[start_index:end_index] = collocation_points
        
        # 创建回调并用当前模型性能初始化
        stopper = EarlyCycleStopper(
            detection_threshold=detection_threshold,
            display_every=5,  # 与 V1 保持一致，提升检测频率
            checkpoint_path_prefix=checkpoint_path_prefix
        )
        
        initial_mre = self.mean_relative_error_metric(None, None)
        
        epochs_before_cycle = self.model.train_state.step or 0
        self.model.save(checkpoint_path_prefix, verbose=0)
        initial_model_path = f"{checkpoint_path_prefix}-{epochs_before_cycle}.pt"
        
        stopper.reset_cycle(initial_mre=initial_mre, initial_model_path=initial_model_path)
        
        print(f"INFO: (PINNModel) Starting training cycle (max: {max_epochs} epochs)...")
        print(f"    Initial MRE for this cycle is {initial_mre:.6f}")

        # 分段训练以便在 detect_every 间隔内检查停滞/快速改善
        remaining_epochs = max_epochs
        stagnation_detected = False
        rollback_event = None
        losshistory = None
        train_state = None

        while remaining_epochs > 0:
            epochs_to_run = min(detect_every, remaining_epochs)
            losshistory, train_state = self.model.train(
                iterations=epochs_to_run,
                callbacks=[stopper],
                display_every=5  # 与 V1 保持一致
            )

            should_exit_cycle = False

            # 停滞检测：当前MRE变差则回退到最佳并提前结束本周期
            if stopper.best_model_path and os.path.exists(stopper.best_model_path):
                latest_mre = self.model.train_state.metrics_test[-1] if self.model.train_state.metrics_test else None
                if latest_mre is not None and latest_mre > stopper.best_mre:
                    print(f"    ⚠️ Stagnation detected: MRE increased to {latest_mre:.6f} (best {stopper.best_mre:.6f}).")
                    stagnation_detected = True
                    self.model.restore(stopper.best_model_path, verbose=0)
                    rollback_event = (self.model.train_state.step or 0, 'rollback')
                    should_exit_cycle = True

            # 快速改善早停：回调标记 should_stop 时提前结束本周期
            if stopper.should_stop:
                should_exit_cycle = True

            if should_exit_cycle:
                break

            remaining_epochs -= epochs_to_run
        
        # 恢复周期内找到的最佳模型并清理临时文件
        if stopper.best_model_path and os.path.exists(stopper.best_model_path):
            print(f"INFO: Restoring best model from cycle: {stopper.best_model_path}")
            self.model.restore(stopper.best_model_path, verbose=1)
            if rollback_event is None:
                rollback_event = (self.model.train_state.step or 0, 'rollback')
            try:
                os.remove(stopper.best_model_path)
            except OSError as e:
                print(f"Warning: could not remove checkpoint {stopper.best_model_path}: {e}")
        
        # 返回训练结果
        final_mre = self.mean_relative_error_metric(None, None)
        print(f"INFO: (PINNModel) Cycle finished. Final MRE: {final_mre:.6f}")
        
        events = stopper.events.copy()
        if rollback_event:
            events.append(rollback_event)
        
        return {
            "stagnation_detected": stagnation_detected,
            "losshistory": losshistory,
            "train_state": train_state,
            "best_mre": stopper.best_mre,
            "final_mre": final_mre,
            "events": events
        }

    def predict(self, points: np.ndarray) -> np.ndarray:
        """使用训练好的PINN模型进行预测，返回物理尺度（线性）的值。"""
        pred_log = self.model.predict(points, operator=None)
        return np.exp(pred_log).flatten()

    def compute_pde_residual(self, points: np.ndarray) -> np.ndarray:
        """计算给定点上的PDE残差的绝对值。"""
        residuals = self.model.predict(points, operator=self.pde)
        return np.abs(residuals).flatten()
