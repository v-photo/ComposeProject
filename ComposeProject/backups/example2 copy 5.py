import numpy as np
import sys
from pathlib import Path
import deepxde as dde
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 路径设置 ---
try:
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent
except NameError:
    project_root = Path('.').resolve()

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'PINN'))
sys.path.insert(0, str(project_root / 'Kriging'))

# --- 模块导入 ---
try:
    from PINN.data_processing import DataLoader
    from PINN.dataAnalysis import get_data
    from myKriging import training as kriging_training, testing as kriging_testing
    print("✅ 外部数据模块导入成功。")
except ImportError as e:
    print(f"❌ 外部数据模块导入失败: {e}")
    sys.exit(1)

# =================================================================================
#  免责声明：以下类和函数均为占位符 (Placeholder)
#  您需要根据您现有的 PINN 和 Kriging 库来填充它们的具体实现。
#  这里的骨架是为了清晰地展示"克里金引导的自适应PINN训练"这一技术路线。
# =================================================================================

class DummyDataLoader:
    """
    一个数据加载器，用于从外部文件加载初始训练数据(替代原有的DummyDataLoader)。
    """
    def __init__(self, data_path: str, space_dims: np.ndarray, num_samples: int):
        self.data_path = data_path
        self.space_dims = space_dims
        self.num_samples = num_samples
        print(f"INFO: (DataLoader) Initialized with data_path='{self.data_path}'")

    def get_training_data(self, split_ratios: list = None):
        """
        加载、处理并采样稀疏训练点，并根据指定的比例列表进行分割。
        
        Args:
            split_ratios (list, optional): 一个浮点数列表，其和应小于1。
                例如 [0.7, 0.1, 0.1] 代表：
                - 70% 作为主训练集
                - 10% 作为第一个储备集
                - 10% 作为第二个储备集
                - 剩余的 10% 将作为测试集。
                如果为 None，则使用默认的 80/20 训练/测试分割。
        """
        # ... (前面加载和采样数据的部分保持不变) ...
        print(f"INFO: (DataLoader) Loading raw data from {self.data_path}...")
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}，请检查路径。")
            
        raw_data = get_data(self.data_path)
        
        print("INFO: (DataLoader) Normalizing dose data...")
        dose_data = DataLoader.load_dose_from_dict(
            data_dict=raw_data,
            space_dims=self.space_dims
        )
        
        print(f"INFO: (DataLoader) Sampling {self.num_samples} training points...")
        train_points, train_values, _ = DataLoader.sample_training_points(
            dose_data, 
            num_samples=self.num_samples,
            sampling_strategy='positive_only',
        )
        print(f"INFO: (DataLoader) ✅ Successfully sampled {len(train_points)} points.")

        # 将坐标和值合并成 [x, y, z, value] 格式
        all_sampled_data = np.hstack([train_points, train_values.reshape(-1, 1)])
        
        # [新逻辑] 使用可配置的分割策略
        if split_ratios is None:
            # 默认行为：80/20 分割
            main_train_set, test_set = train_test_split(all_sampled_data, test_size=0.2, random_state=42)
            reserve_pools = []
        else:
            if sum(split_ratios) >= 1.0:
                raise ValueError("split_ratios 的总和必须小于 1.0，以便为测试集留出空间。")

            remaining_data = all_sampled_data
            data_pools = []
            
            # 循环切分出主训练集和所有储备集
            current_total_fraction = 1.0
            for ratio in split_ratios:
                # 计算当前比例相对于剩余数据量的比例
                split_fraction = ratio / current_total_fraction
                pool, remaining_data = train_test_split(remaining_data, test_size=(1.0 - split_fraction), random_state=42)
                data_pools.append(pool)
                current_total_fraction -= ratio

            main_train_set = data_pools[0]
            reserve_pools = data_pools[1:]
            test_set = remaining_data # 剩下的所有数据都作为测试集
        
        print(f"INFO: (DataLoader) ✅ Split data into: Main training ({len(main_train_set)}), Test ({len(test_set)}), Reserve Pools ({len(reserve_pools)} pools).")
        if reserve_pools:
            for i, pool in enumerate(reserve_pools):
                print(f"    - Reserve Pool {i+1}: {len(pool)} points")

        return main_train_set, reserve_pools, test_set, dose_data

class GPUKriging:
    """
    [真实实现] GPU加速的克里金模型的适配器。
    该实现借鉴了 ComposeTools.py 中的 KrigingAdapter，并调用了 myKriging 库。
    """
    def __init__(self, variogram_model='exponential', **kwargs):
        """
        初始化Kriging适配器。
        
        Args:
            variogram_model (str): 克里金所需的变异函数模型。
            **kwargs: 其他可以传递给 myKriging 库的参数 (如 nlags, block_size)。
        """
        self.model = None
        self._is_fitted = False
        self.variogram_model = variogram_model
        self.kriging_params = kwargs
        print(f"INFO: (GPUKriging) Initialized with variogram model: {self.variogram_model}")

    def fit(self, points: np.ndarray, values: np.ndarray):
        """
        使用稀疏的点坐标和对应的残差值来训练克里金代理模型。
        """
        print(f"INFO: (GPUKriging) Fitting model with {len(points)} points...")
        # 1. 将 NumPy 数组转换为 myKriging 所期望的 Pandas DataFrame
        df = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'target': values
        })

        # 2. 调用外部的 kriging_training 函数
        self.model = kriging_training(
            df=df,
            variogram_model=self.variogram_model,
            nlags=self.kriging_params.get('nlags', 8),
            enable_plotting=False, # 训练代理模型时通常不绘图
            weight=False,
            uk=False,
            cpu_on=False # 确保使用GPU
        )
        
        self._is_fitted = True
        print("INFO: (GPUKriging) ✅ Model fitted.")

    def predict(self, points_to_predict: np.ndarray) -> np.ndarray:
        """
        对新的点进行批量预测，利用GPU加速。
        """
        if not self._is_fitted:
            raise RuntimeError("Kriging model must be fitted before prediction.")
        
        print(f"INFO: (GPUKriging) Predicting values for {len(points_to_predict)} points...")
        # 1. 将 NumPy 数组转换为 myKriging 所期望的 Pandas DataFrame
        df_pred = pd.DataFrame({
            'x': points_to_predict[:, 0],
            'y': points_to_predict[:, 1],
            'z': points_to_predict[:, 2],
            'target': np.zeros(points_to_predict.shape[0]) # 虚拟目标值
        })

        # 2. 调用外部的 kriging_testing 函数，确保使用GPU加速配置
        predictions, _ = kriging_testing(
            df=df_pred,
            model=self.model,
            block_size=self.kriging_params.get('block_size', 10000),
            cpu_on=False, # 确保使用GPU
            style="gpu_b", # 使用GPU批处理风格
            multi_process=False,
            print_time=False,
            torch_ac=False, # 使用PyTorch加速
            compute_precision=False # 预测残差时不需要精度计算
        )
        
        print(f"INFO: (GPUKriging) ✅ Prediction complete.")
        return predictions.flatten() # 确保返回一维数组

class PINNModel:
    """
    [真实实现] 物理信息神经网络（PINN）。
    该实现被设计为可从外部控制的模式，以支持自适应训练流程。
    """
    def __init__(self, dose_data: dict, training_data: np.ndarray, test_data: np.ndarray, num_collocation_points: int, network_layers=[3, 64, 64, 64, 1], lr=1e-3):
        """
        初始化PINN模型，但与PINNTrainer不同，这里只做准备工作，不开始训练。
        
        Args:
            dose_data (dict): 从DataLoader加载的数据字典。
            training_data (np.ndarray): 稀疏训练数据 [x,y,z,value]。
            test_data (np.ndarray): 稀疏测试数据 [x,y,z,value]。
            num_collocation_points (int): 求解域点的数量。
            network_layers (list): 神经网络结构。
            lr (float): 学习率。
        """
        print("INFO: (PINNModel) Initializing a DeepXDE-based model for external control...")
        
        self.test_data_linear = test_data # 存储线性尺度的测试数据
        
        # 1. 定义几何
        world_min = dose_data['world_min']
        world_max = dose_data['world_max']
        self.geometry = dde.geometry.Cuboid(world_min, world_max)

        # 2. 定义可训练参数
        k_initial_guess = 1.0 
        self.log_k_pinn = dde.Variable(np.log(k_initial_guess))
        
        # 3. 定义PDE为类的方法，以便在其他地方复用
        self.pde = self._build_pde_func()
        
        # 4. 定义训练数据点
        observe_x = training_data[:, :3]
        observe_y = np.log(np.maximum(training_data[:, 3:], 1e-30))
        data_points = dde.icbc.PointSetBC(observe_x, observe_y, component=0)
        
        # 5. 组合成dde.data.PDE对象
        self.data = dde.data.PDE(
            self.geometry,
            self.pde,
            [data_points],
            num_domain=num_collocation_points,
            anchors=observe_x,
            # 我们自定义的指标会使用我们自己存储的self.test_data_linear
        )
        
        # 6. 创建网络和模型
        self.net = dde.nn.FNN(network_layers, "tanh", "Glorot normal")
        self.model = dde.Model(self.data, self.net)
        
        # 7. 自定义指标函数被移出为类方法 mean_relative_error_metric

        self.lr = lr # 保存学习率以备重编译时使用
        
        # [新增] MRE历史记录列表
        self.mre_history = []
        self.epoch_history = []
        
        # 8. 编译模型，加入自定义指标
        self.compile_model()
        print("INFO: (PINNModel) ✅ Model compiled and ready for training cycles.")
        
    def compile_model(self):
        """将模型编译封装成一个方法，方便重用。"""
        # [修正] 在这里设置指标函数的显示名称
        # self.mean_relative_error_metric.__name__ = "MRE_test_set" # [修正] 移除此行，不能为类方法设置__name__

        self.model.compile(
            "adam", 
            lr=self.lr, 
            loss_weights=[1, 10], 
            external_trainable_variables=[self.log_k_pinn],
            metrics=[self.mean_relative_error_metric] # [修正] 传递函数对象，而不是字符串
        )

    def mean_relative_error_metric(self, y_true_ignored, y_pred_ignored):
        """
        一个"hack"的指标函数。它忽略dde传入的参数，
        转而使用我们自己存储的、基于真实物理值的测试集进行评估。
        """
        # 使用模型对我们自己的测试点进行预测
        test_x = self.test_data_linear[:, :3]
        pred_y_log = self.model.predict(test_x)
        
        # 将预测值和真实值都转换回线性物理尺度
        pred_y_linear = np.exp(pred_y_log)
        true_y_linear = self.test_data_linear[:, 3:]
        
        # 计算相对误差
        mre = np.mean(np.abs(true_y_linear - pred_y_linear) / (true_y_linear + 1e-10))
        
        # [新增] 记录MRE历史
        current_epoch = self.model.train_state.step if self.model.train_state.step else 0
        self.mre_history.append(mre)
        self.epoch_history.append(current_epoch)
        
        return mre

    def inject_new_data(self, new_data_array: np.ndarray):
        """
        [新能力] 向模型中注入新的训练数据点。
        """
        print(f"\nINFO: (PINNModel)  injecting {len(new_data_array)} new data points...")
        
        # 1. 获取现有数据
        current_bc = self.data.bcs[0]
        current_points = current_bc.points
        current_values_log = current_bc.values.cpu()

        # 2. 准备新数据
        new_points = new_data_array[:, :3]
        new_values_log = np.log(np.maximum(new_data_array[:, 3:], 1e-30)).reshape(-1, 1)

        # 3. 合并新旧数据
        combined_points = np.vstack([current_points, new_points])
        combined_values_log = np.vstack([current_values_log, new_values_log])
        
        print(f"    Total training points increased to {len(combined_points)}.")

        # 4. 创建新的 PointSetBC 和 PDE 数据对象
        new_bc = dde.icbc.PointSetBC(combined_points, combined_values_log, component=0)
        
        # 更新锚点以包含所有训练数据
        new_anchors = combined_points
        
        new_data_obj = dde.data.PDE(
            self.geometry,
            self.pde,
            [new_bc],
            num_domain=self.data.num_domain,
            anchors=new_anchors
        )
        
        # 5. 更新模型的数据并重新编译
        self.data = new_data_obj
        self.model.data = self.data
        self.compile_model()
        print("INFO: (PINNModel) ✅ Model re-compiled with new data. Initializing new train state...")
        # [修正] 调用 train(0) 来强制使用新数据对象重建训练状态
        self.model.train(iterations=0, display_every=100000) # display_every设为大数以避免不必要的输出
        print("INFO: (PINNModel) ✅ New train state initialized.")
        
        # [修正] 注入数据后，记录一次当前MRE以保持历史连续性
        test_x = self.test_data_linear[:, :3]
        pred_y_log = self.model.predict(test_x)
        pred_y_linear = np.exp(pred_y_log)
        true_y_linear = self.test_data_linear[:, 3:]
        current_mre = np.mean(np.abs(true_y_linear - pred_y_linear) / (true_y_linear + 1e-10))
        current_epoch = self.model.train_state.step if self.model.train_state.step else 0
        self.mre_history.append(current_mre)
        self.epoch_history.append(current_epoch)
        print(f"INFO: (PINNModel) MRE after data injection: {current_mre:.6f} at epoch {current_epoch}")
        
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

    def run_training_cycle(self, max_epochs: int, detect_every: int, collocation_points: np.ndarray, 
                         detection_threshold: float = 0.1):
        """
        [重构] 执行一个带有动态停止条件的训练周期。
        
        Args:
            max_epochs (int): 当前周期的最大训练轮数。
            detect_every (int): 每隔多少轮进行一次性能检测。
            collocation_points (np.ndarray): 用于本周期的配置点。
            detection_threshold (float): 触发早停的相对改进阈值。
        
        Returns:
            dict: 一个包含训练结果信息的字典，如{'stagnation_detected': bool}
        """
        # 1. 使用绝对路径定义检查点文件名的前缀，并确保目录存在
        script_dir = Path(__file__).parent.resolve()
        checkpoint_path_prefix = str(script_dir / "models" / "best_model_in_cycle")
        os.makedirs(Path(checkpoint_path_prefix).parent, exist_ok=True)

        # 2. 更新求解域点
        num_bc_points = self.data.bcs[0].points.shape[0]
        if self.model.train_state.X_train is None:
            # 初始化训练状态，以便可以修改X_train
            self.model.train(iterations=0)
        start_index = num_bc_points
        end_index = len(self.model.train_state.X_train) - len(self.data.anchors)
        self.model.train_state.X_train[start_index:end_index] = collocation_points
        
        # [新] 初始化本周期的返回状态
        stagnation_detected_this_run = False
        data_injected_this_cycle = False
        
        # 3. 创建回调，并用当前模型的性能初始化它
        stopper = EarlyCycleStopper(
            detection_threshold=detection_threshold,
            display_every=5,
            checkpoint_path_prefix=checkpoint_path_prefix
        )
        # [修正] 在重置时，传入当前模型的MRE和初始检查点作为基线
        test_x = self.test_data_linear[:, :3]
        pred_y_log = self.model.predict(test_x)
        pred_y_linear = np.exp(pred_y_log)
        true_y_linear = self.test_data_linear[:, 3:]
        initial_mre = np.mean(np.abs(true_y_linear - pred_y_linear) / (true_y_linear + 1e-10))

        # 为初始状态创建第一个基准检查点
        epochs_before_cycle = self.model.train_state.step or 0
        self.model.save(checkpoint_path_prefix, verbose=0)
        initial_model_path = f"{checkpoint_path_prefix}-{epochs_before_cycle}.pt"
        
        stopper.reset_cycle(initial_mre=initial_mre, initial_model_path=initial_model_path)
        
        print(f"INFO: (PINNModel) Starting dynamic training cycle (max: {max_epochs} epochs, detect every: {detect_every})...")
        print(f"    Initial MRE for this cycle is {initial_mre:.4f}")
        
        remaining_epochs = max_epochs
        while remaining_epochs > 0:
            epochs_to_run = min(detect_every, remaining_epochs)
            
            self.model.train(
                iterations=epochs_to_run, 
                display_every=5,
                callbacks=[stopper]
            )

            # --- [新策略] 检查是否需要提前结束本轮自适应周期 ---
            should_exit_cycle = False

            # 条件1: 停滞 (Stagnation) - 模型性能在本轮训练后变差
            if stopper.best_model_path and os.path.exists(stopper.best_model_path):
                latest_mre = self.model.train_state.metrics_test[-1]
                if latest_mre > stopper.best_mre:
                    print(f"    ⚠️ Stagnation detected: MRE increased to {latest_mre:.4f} (best is {stopper.best_mre:.4f}).")
                    stagnation_detected_this_run = True
                    
                    print(f"    ↳ Forcing new adaptive resampling cycle...")
                    self.model.restore(stopper.best_model_path, verbose=0) 
                    should_exit_cycle = True

            # 条件2: 快速提升 (Rapid Improvement) - 我们的旧早停逻辑
            if stopper.should_stop:
                print(f"\nINFO: (PINNModel) 📈 Rapid improvement! Capitalizing on gains and forcing new resampling.")
                should_exit_cycle = True
            
            if should_exit_cycle:
                break
                
            remaining_epochs -= epochs_to_run
        else:
             print(f"\nINFO: (PINNModel) Max epochs reached for this cycle.")

        # 4. [重要] 无论如何，都从最终的最佳检查点恢复模型，并清理文件
        if stopper.best_model_path and os.path.exists(stopper.best_model_path):
            print(f"INFO: (PINNModel) Restoring model to best state from '{stopper.best_model_path}'...")
            self.model.restore(stopper.best_model_path, verbose=1)
            os.remove(stopper.best_model_path) # 清理临时文件
        else:
            print("WARNING: (PINNModel) Best checkpoint file not found. Model may not be in its best state.")
            
        return {'stagnation_detected': stagnation_detected_this_run}

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        [新] 使用训练好的模型进行预测，并返回线性尺度的物理值。
        
        Args:
            points (np.ndarray): 待预测点的坐标，形状为 (N, 3)。
            
        Returns:
            np.ndarray: 预测的物理值，形状为 (N,)。
        """
        print(f"INFO: (PINNModel) Predicting on {len(points)} points...")
        # model.predict 返回的是对数尺度的值
        pred_y_log = self.model.predict(points)
        # 转换回线性物理尺度
        pred_y_linear = np.exp(pred_y_log)
        return pred_y_linear.flatten()

    def compute_pde_residual(self, points: np.ndarray) -> np.ndarray:
        """
        [真实实现] 计算给定点上的物理方程残差。
        利用 deepxde 的 model.predict(operator=...) 功能。
        """
        print(f"INFO: (PINNModel) Computing PDE residuals for {len(points)} points...")
        
        # deepxde.Model.predict 可以接受一个 operator 参数
        # 我们将 self.pde (在__init__中定义的函数) 作为算子传入
        residuals = self.model.predict(points, operator=self.pde)
        
        # 返回残差的绝对值，并展平为一维数组
        return np.abs(residuals).flatten()

class EarlyCycleStopper(dde.callbacks.Callback):
    """
    一个自定义回调，用于在训练周期内实现基于性能的"早停"。
    同时自己负责保存周期内的最佳模型。
    """
    def __init__(self, detection_threshold: float, display_every: int, checkpoint_path_prefix: str):
        super().__init__()
        self.threshold = detection_threshold
        self.display_every = display_every
        self.checkpoint_path_prefix = checkpoint_path_prefix
        self.best_mre = np.inf
        self.should_stop = False
        self.best_model_path = "" # 将存储最佳模型的完整真实路径

    def reset_cycle(self, initial_mre: float = np.inf, initial_model_path: str = ""):
        """
        手动重置整个周期的状态，为新的自适应周期做准备。
        可以接收一个初始MRE和模型路径作为本周期的性能基线。
        """
        # 清理上一轮可能遗留的检查点文件
        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
        
        self.best_mre = initial_mre
        self.best_model_path = initial_model_path
        self.should_stop = False

    def on_epoch_end(self):
        """在每个 epoch 结束时被调用, 并且在这里检查性能"""
        if self.model.train_state.step > 0 and self.model.train_state.step % self.display_every == 0:
            if not self.model.train_state.metrics_test:
                 return

            latest_mre = self.model.train_state.metrics_test[-1]
            
            if self.best_mre != np.inf:
                improvement = self.best_mre - latest_mre
                required_improvement_amount = self.best_mre * self.threshold
                
                if improvement > required_improvement_amount:
                    print(f"    💡 Early Stop: MRE dropped from {self.best_mre:.4f} to {latest_mre:.4f} (>{self.threshold:.0%}).")
                    self.should_stop = True
            
            # 判断当前模型是否是新的最优模型，如果是，则保存它
            if latest_mre < self.best_mre:
                print(f"    ⭐ New best model found (MRE: {latest_mre:.4f}). Checkpointing...")
                self.best_mre = latest_mre

                # 清理上一个最佳模型
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                
                # 构建新的最佳模型路径并保存
                current_step = self.model.train_state.step
                self.best_model_path = f"{self.checkpoint_path_prefix}-{current_step}.pt"
                self.model.save(self.checkpoint_path_prefix, verbose=0)

class AdaptiveSampler:
    """
    [建议您实现] 自适应采样器。
    """
    def __init__(self, domain_bounds, total_candidates=100000):
        self.bounds = domain_bounds
        # 预先在整个域内生成大量的候选点，后续从中筛选
        self.candidate_points = np.random.rand(total_candidates, 3) * \
            (domain_bounds[1] - domain_bounds[0]) + domain_bounds[0]
        print(f"INFO: (AdaptiveSampler) Initialized with {total_candidates} candidate points.")

    def generate_new_collocation_points(
        self,
        kriging_model: GPUKriging,
        num_points_to_sample: int,
        force_exploration: bool = False
    ) -> np.ndarray:
        """
        使用Kriging模型引导生成新的配置点。
        Args:
            kriging_model: 训练好的残差代理模型。
            num_points_to_sample: 需要生成的总点数。
            force_exploration: 是否因模型停滞而强制增加探索比例。
        Returns:
            np.ndarray: 新的配置点集。
        """
        # [新逻辑] 根据是否停滞来动态调整探索率
        if force_exploration:
            exploration_ratio = 0.3 # 停滞时，大幅增加随机探索
            print(f"INFO: (AdaptiveSampler) Stagnation detected! Increasing exploration ratio to {exploration_ratio:.0%}.")
        else:
            exploration_ratio = 0.1 # 正常情况下的探索率

        # 1. 使用Kriging代理模型预测所有候选点的残差
        predicted_residuals = kriging_model.predict(self.candidate_points)
        print(f"    - Kriging预测残差统计 (在 {len(self.candidate_points)} 个候选点上):")
        print(f"      - Max={np.max(predicted_residuals):.4e}, "
              f"Min={np.min(predicted_residuals):.4e}, "
              f"Mean={np.mean(predicted_residuals):.4e}, "
              f"Std={np.std(predicted_residuals):.4e}")

        # 2. "Hard-Case Mining": 找到预测残差最大的点的索引
        num_exploitation_points = int(num_points_to_sample * (1 - exploration_ratio))
        hard_case_indices = np.argsort(predicted_residuals)[-num_exploitation_points:]
        exploitation_points = self.candidate_points[hard_case_indices]

        # 3. "Exploration": 加入一部分随机点以避免陷入局部最优
        num_exploration_points = num_points_to_sample - num_exploitation_points
        random_indices = np.random.choice(len(self.candidate_points), num_exploration_points, replace=False)
        exploration_points = self.candidate_points[random_indices]

        print(f"INFO: (AdaptiveSampler) Generated {num_exploitation_points} exploitation points and {num_exploration_points} exploration points.")
        
        return np.vstack([exploitation_points, exploration_points])

def main():
    """
    主函数，编排整个"克里金引导的自适应PINN训练"流程。
    """
    print("\n" + "="*60)
    print("🚀 开始执行：克里金引导的自适应PINN训练")
    print("="*60 + "\n")

    # --- 1. 初始化 ---
    # !! 注意: DOMAIN_BOUNDS 现在仅用于可视化或采样器，实际物理边界由加载的数据决定 !!
    DOMAIN_BOUNDS = np.array([[0., 0., 0.], [1., 1., 1.]]) 
    TOTAL_EPOCHS = 2400
    ADAPTIVE_CYCLE_EPOCHS = 400  # 每多少个epoch执行一次自适应调整
    DETECT_EPOCHS = 100 # 每50轮检测一次性能 [修正注释]
    DATA_SPLIT_RATIOS = [0.7] + [0.05]*6
    
    # --- 数据加载参数 ---
    DATA_PATH = "PINN/DATA.xlsx"
    SPACE_DIMS = np.array([20.0, 10.0, 10.0])
    NUM_SAMPLES = 50
    
    # --- 模型训练参数 ---
    NUM_COLLOCATION_POINTS = 4096
    NUM_RESIDUAL_SCOUT_POINTS = 5000 # 用于侦察的点数，远少于训练点数

    # 1. 数据加载
    data_loader = DummyDataLoader(
        data_path=DATA_PATH,
        space_dims=SPACE_DIMS,
        num_samples=NUM_SAMPLES
    )
    main_train_set, reserve_data_pools, test_data, dose_data = data_loader.get_training_data(
        split_ratios=DATA_SPLIT_RATIOS
    )

    # 2. 模型和采样器初始化
    world_min = dose_data['world_min']
    world_max = dose_data['world_max']
    
    pinn = PINNModel(
        dose_data=dose_data, 
        training_data=main_train_set, # [新] 只用主训练集初始化
        test_data=test_data,
        num_collocation_points=NUM_COLLOCATION_POINTS
    )
    kriging = GPUKriging()
    sampler = AdaptiveSampler(domain_bounds=np.vstack([world_min, world_max]))
    
    # 初始配置点：在真实物理空间内采样
    current_collocation_points = (np.random.rand(NUM_COLLOCATION_POINTS, 3) * 
                                  (world_max - world_min) + world_min)

    # --- 3. [修正] 训练循环 ---
    # 使用 while 循环来确保总训练轮数达标
    total_epochs_trained = 0
    consecutive_stagnation_count = 0 # [新] 连续停滞计数器
    
    # [新增] 重要事件记录列表，用于图表标注
    important_events = []  # 格式: [(epoch, event_type, description), ...]

    while total_epochs_trained < TOTAL_EPOCHS:
        remaining_total_epochs = TOTAL_EPOCHS - total_epochs_trained
        
        # 本次自适应周期的最大训练轮数，不能超过总剩余轮数
        cycle_max_epochs = min(ADAPTIVE_CYCLE_EPOCHS, remaining_total_epochs)
        
        print(f"\n--- 主循环周期: 目标训练 {total_epochs_trained} -> {total_epochs_trained + cycle_max_epochs} ---")

        # 2a. 使用当前的配置点，对PINN进行一轮常规训练
        print(f"PHASE 2a: 常规PINN训练 (本周期上限: {cycle_max_epochs} epochs)...")
        
        # 记录进入此周期前的训练步数
        epochs_before_cycle = pinn.model.train_state.step or 0
        
        # 调用改造后的方法，并接收其返回结果
        cycle_result = pinn.run_training_cycle(
            max_epochs=cycle_max_epochs,
            detect_every=DETECT_EPOCHS,
            collocation_points=current_collocation_points,
            detection_threshold=0.1
        )
        
        # 计算本周期实际训练了多少轮
        epochs_this_cycle = (pinn.model.train_state.step or 0) - epochs_before_cycle
        total_epochs_trained += epochs_this_cycle

        # --- [新逻辑] ---
        # 1. 更新停滞计数器
        stagnation_this_cycle = cycle_result.get('stagnation_detected', False)
        if stagnation_this_cycle:
            consecutive_stagnation_count += 1
            print(f"INFO: Consecutive stagnation count increased to: {consecutive_stagnation_count}")
        else:
            # 任何成功的周期都会重置计数器
            if consecutive_stagnation_count > 0:
                print(f"INFO: Training successful, resetting stagnation count from {consecutive_stagnation_count} to 0.")
            consecutive_stagnation_count = 0
        
        print(f"\nINFO: 本周期实际训练 {epochs_this_cycle} 轮. 总训练进度: {total_epochs_trained}/{TOTAL_EPOCHS}")
        
        # 如果已经训练够了，就提前结束主循环
        if total_epochs_trained >= TOTAL_EPOCHS:
            print("\nINFO: 总训练轮数已达到目标，结束自适应训练。")
            break

        # 2. 检查是否需要执行干预 (注入数据 + 克里金重采样)
        if consecutive_stagnation_count >= 2:
            print("\n" + "!"*60)
            print("!! 连续停滞两次，触发干预机制 !!")
            print("!"*60)

            # --- 干预措施 1: 注入新数据 (如果还有) ---
            if reserve_data_pools:
                print("\nPHASE A: 注入新的储备训练数据...")
                data_injection_epoch = pinn.model.train_state.step or 0
                data_to_inject = reserve_data_pools.pop(0)
                pinn.inject_new_data(data_to_inject)
                print("PHASE A: ✅ 新数据注入完成。")
                
                # [新增] 记录数据注入事件
                important_events.append((
                    data_injection_epoch, 
                    'data_injection', 
                    f'数据注入 (+{len(data_to_inject)}点)'
                ))
            else:
                print("\nWARNING: 已无更多储备数据可注入。")

            # --- 干预措施 2: 克里金引导的自适应采样 ---
            print("\nPHASE B: 开始克里金引导的自适应采样...")
            
            # 残差"侦察"
            scout_points = (np.random.rand(NUM_RESIDUAL_SCOUT_POINTS, 3) *
                            (world_max - world_min) + world_min)
            true_residuals = pinn.compute_pde_residual(scout_points)
            print(f"    - 真实PDE残差统计 (在 {len(scout_points)} 个侦察点上):")
            print(f"      - Max={np.max(true_residuals):.4e}, "
                  f"Min={np.min(true_residuals):.4e}, "
                  f"Mean={np.mean(true_residuals):.4e}, "
                  f"Std={np.std(true_residuals):.4e}")
            
            # 克里金代理建模
            kriging.fit(scout_points, true_residuals)

            # 自适应采样
            kriging_epoch = pinn.model.train_state.step or 0
            num_collocation_to_generate = pinn.data.num_domain
            print(f"INFO: Dynamically calculated {num_collocation_to_generate} collocation points to generate.")

            current_collocation_points = sampler.generate_new_collocation_points(
                kriging_model=kriging,
                num_points_to_sample=num_collocation_to_generate,
                force_exploration=True  # 因为停滞了，所以强制探索
            )
            print("PHASE B: ✅ 新的自适应配置点已生成。")
            
            # [新增] 记录克里金应用事件
            important_events.append((
                kriging_epoch, 
                'kriging_resampling', 
                '克里金引导重采样'
            ))

            # --- 干预后重置计数器 ---
            print("\nINFO: 干预措施已执行，重置停滞计数器。")
            consecutive_stagnation_count = 0
            print("-" * 60)
        
        else:
            # 如果没有达到干预阈值，则不执行克里金采样，继续使用现有配置点
            print("\nINFO: 未达到干预阈值，下一周期将继续使用当前配置点。")
            # 在这种情况下，我们不需要更新 current_collocation_points
            pass

    print("\n" + "="*60)
    print("🎉 训练完成!")
    print("="*60 + "\n")

    # --- 4. 训练原始PINN作为对比模型 ---
    print("\n" + "="*60)
    print("🚀 开始训练原始PINN作为对比基线")
    print("="*60 + "\n")

    # [新] 为基线模型准备完整的训练数据
    # 直接将第一次加载时分割好的所有数据块合并起来
    all_training_blocks = [main_train_set] + reserve_data_pools
    full_training_data = np.vstack(all_training_blocks)

    pinn_baseline = PINNModel(
        dose_data=dose_data, 
        training_data=full_training_data, # [新] 使用全部训练数据
        test_data=test_data,
        num_collocation_points=NUM_COLLOCATION_POINTS
    )
    # 为原始PINN生成一次性的、固定的配置点
    baseline_collocation_points = (np.random.rand(NUM_COLLOCATION_POINTS, 3) * 
                                   (world_max - world_min) + world_min)

    # 原始PINN使用固定的训练周期
    print("INFO: (Baseline PINN) Setting collocation points...")
    # [修正] 将生成的配置点手动设置到模型中
    num_bc_points_base = pinn_baseline.data.bcs[0].points.shape[0]
    if pinn_baseline.model.train_state.X_train is None:
        pinn_baseline.model.train(iterations=0)
    start_index_base = num_bc_points_base
    end_index_base = len(pinn_baseline.model.train_state.X_train) - len(pinn_baseline.data.anchors)
    pinn_baseline.model.train_state.X_train[start_index_base:end_index_base] = baseline_collocation_points

    print("INFO: (Baseline PINN) Starting training...")
    pinn_baseline.model.train(iterations=TOTAL_EPOCHS, display_every=5)
    
    print("\n" + "="*60)
    print("🎉 原始PINN训练完成!")
    print("="*60 + "\n")

    # --- 5. 最终结果评估与对比 ---
    print("\n" + "="*60)
    print("📊 最终结果评估与对比")
    print("="*60 + "\n")

    test_points = test_data[:, :3]
    true_values = test_data[:, 3]

    print("正在评估两个模型的最终性能...")
    adaptive_preds = pinn.predict(test_points)
    baseline_preds = pinn_baseline.predict(test_points)

    def calculate_mre(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-10))

    mre_adaptive = calculate_mre(true_values, adaptive_preds)
    mre_baseline = calculate_mre(true_values, baseline_preds)

    print(f"\n{'模型':<28} | {'平均相对误差 (MRE) on Test Set':<30}")
    print("-" * 65)
    print(f"{'Kriging引导的自适应PINN':<28} | {mre_adaptive:<30.6%}")
    print(f"{'原始PINN (固定采样)':<28} | {mre_baseline:<30.6%}")
    print("-" * 65)
    
    # 输出重要事件摘要
    if important_events:
        print(f"\n📋 训练过程重要事件摘要:")
        print("-" * 50)
        for epoch, event_type, description in important_events:
            event_name = "🔄 克里金重采样" if event_type == 'kriging_resampling' else "📊 数据注入"
            print(f"  Epoch {epoch:4d}: {event_name} - {description}")
        print("-" * 50)

    # --- 6. 绘制对比图 ---
    print("\n" + "="*60)
    print("📈 绘制训练过程MRE对比图")
    print("="*60 + "\n")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 打印将要标注的事件
    if important_events:
        print(f"INFO: 准备在图表中标注 {len(important_events)} 个重要时间点:")
        for epoch, event_type, description in important_events:
            print(f"  - Epoch {epoch}: {description}")
    
    # 绘制自适应PINN的MRE历史
    if pinn.epoch_history and pinn.mre_history:
        ax.plot(pinn.epoch_history, pinn.mre_history, 
                label='Kriging引导的自适应PINN', linewidth=2, alpha=0.8, color='blue')
    
    # 绘制基线PINN的MRE历史
    if pinn_baseline.epoch_history and pinn_baseline.mre_history:
        ax.plot(pinn_baseline.epoch_history, pinn_baseline.mre_history, 
                label='原始PINN (固定采样)', linewidth=2, alpha=0.8, color='red')
    
    # 添加重要事件标注
    if important_events:
        print(f"INFO: 标注 {len(important_events)} 个重要时间点...")
        
        # 定义事件类型的颜色和样式
        event_styles = {
            'data_injection': {'color': 'green', 'linestyle': '--', 'alpha': 0.7},
            'kriging_resampling': {'color': 'orange', 'linestyle': '-.', 'alpha': 0.7}
        }
        
        for i, (epoch, event_type, description) in enumerate(important_events):
            style = event_styles.get(event_type, {'color': 'gray', 'linestyle': '-', 'alpha': 0.5})
            
            # 绘制垂直线
            ax.axvline(x=epoch, **style, linewidth=2)
            
            # 获取当前y轴范围来定位文本
            y_min, y_max = ax.get_ylim()
            y_pos = y_max * (0.8 - (i % 3) * 0.15)  # 错开标注位置避免重叠
            
            # 添加文本标注
            ax.annotate(
                f'{description}\n(Epoch {epoch})',
                xy=(epoch, y_pos),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=style['color'], alpha=0.3),
                fontsize=9,
                ha='left'
            )
        
        # 添加图例说明
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', linestyle='--', label='数据注入'),
            Line2D([0], [0], color='orange', linestyle='-.', label='克里金重采样')
        ]
        
        # 创建第二个图例
        second_legend = ax.legend(handles=legend_elements, loc='upper right', 
                                 fontsize=10, title='重要事件', title_fontsize=11)
        ax.add_artist(second_legend)  # 保持原有图例
    
    # 设置图表属性
    ax.set_xlabel('训练轮数 (Epochs)', fontsize=12)
    ax.set_ylabel('平均相对误差 (MRE)', fontsize=12)
    ax.set_title('自适应PINN vs 基线PINN: 训练过程MRE对比', fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=11)  # 调整原图例位置
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # 使用对数坐标更好地显示误差变化
    
    # 保存图表
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "mre_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "mre_comparison.pdf", bbox_inches='tight')
    
    print(f"✅ 对比图已保存到: {output_dir}")
    print(f"   - PNG格式: {output_dir / 'mre_comparison.png'}")
    print(f"   - PDF格式: {output_dir / 'mre_comparison.pdf'}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main() 