import numpy as np
import time
import sys
from pathlib import Path
import torch
import deepxde as dde
import pandas as pd

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

    def get_training_data(self) -> np.ndarray:
        """
        加载、处理并采样稀疏训练点。
        这些点在PINN中充当"数据真值"(Ground Truth)，是数据损失项的来源。
        其功能类似于原get_boundary_conditions，但数据源是外部文件。
        """
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
        
        # 将采样数据80/20分割为训练集和测试集
        np.random.shuffle(all_sampled_data)
        split_index = int(0.8 * len(all_sampled_data))
        training_data_array = all_sampled_data[:split_index]
        test_data_array = all_sampled_data[split_index:]
        
        print(f"INFO: (DataLoader) ✅ Split data into {len(training_data_array)} training points and {len(test_data_array)} test points.")
        
        # 同时返回包含边界等元数据的 dose_data 字典
        return training_data_array, test_data_array, dose_data

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
        
        # 7. 定义自定义指标函数
        def mean_relative_error_metric(y_true_ignored, y_pred_ignored):
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
            return np.mean(np.abs(true_y_linear - pred_y_linear) / (true_y_linear + 1e-10))

        mean_relative_error_metric.__name__ = "MRE_test_set"

        # 8. 编译模型，加入自定义指标
        self.model.compile(
            "adam", 
            lr=lr, 
            loss_weights=[1, 10], 
            external_trainable_variables=[self.log_k_pinn],
            metrics=[mean_relative_error_metric]
        )
        print("INFO: (PINNModel) ✅ Model compiled and ready for training cycles.")
        
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

    def run_training_cycle(self, epochs: int, collocation_points: np.ndarray):
        """
        [新] 执行一个完整的训练周期。
        这个方法取代了旧的 train_step，以允许deepxde打印训练状态。
        """
        # 1. 更新求解域点 (Collocation Points)
        num_bc_points = self.data.bcs[0].points.shape[0]
        if self.model.train_state.X_train is None:
             self.model.train(iterations=0)
        start_index = num_bc_points
        end_index = len(self.model.train_state.X_train) - len(self.data.anchors)
        self.model.train_state.X_train[start_index:end_index] = collocation_points
        
        # 2. 调用 deepxde 的 train 函数进行指定次数的迭代
        print(f"INFO: (PINNModel) Starting training cycle for {epochs} epochs...")
        self.model.train(iterations=epochs, display_every=100)

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

class AdaptiveSampler:
    """
    [真实实现] 自适应采样器。
    用于根据Kriging的预测结果生成新的训练点。
    """
    def __init__(self, world_min: np.ndarray, world_max: np.ndarray, total_candidates=100000):
        """
        初始化自适应采样器。

        Args:
            world_min (np.ndarray): 真实物理空间的最小坐标。
            world_max (np.ndarray): 真实物理空间的最大坐标。
            total_candidates (int): 在整个域内生成的候选点数量。
        """
        self.world_min = world_min
        self.world_max = world_max
        
        # 在整个真实物理域内生成大量的候选点，后续从中筛选
        self.candidate_points = (np.random.rand(total_candidates, 3) * 
                                 (self.world_max - self.world_min) + self.world_min)
                                 
        print(f"INFO: (AdaptiveSampler) Initialized with {total_candidates} candidate points "
              f"within physical bounds [{world_min.round(2)}, {world_max.round(2)}].")

    def generate_new_collocation_points(
        self,
        kriging_model: GPUKriging,
        num_points_to_sample: int,
        exploration_ratio: float = 0.1
    ) -> np.ndarray:
        """
        使用Kriging模型引导生成新的配置点。
        Args:
            kriging_model: 训练好的残差代理模型。
            num_points_to_sample: 需要生成的总点数。
            exploration_ratio: 从总点数中分出多少比例用于随机探索。
        Returns:
            np.ndarray: 新的配置点集。
        """
        # 1. 使用Kriging代理模型预测所有候选点的残差
        predicted_residuals = kriging_model.predict(self.candidate_points)

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
    TOTAL_EPOCHS = 5000
    ADAPTIVE_CYCLE_EPOCHS = 1000  # 每多少个epoch执行一次自适应调整
    
    # --- 数据加载参数 ---
    DATA_PATH = "PINN/DATA.xlsx"
    SPACE_DIMS = np.array([20.0, 10.0, 10.0])
    NUM_SAMPLES = 300
    
    # --- 模型训练参数 ---
    NUM_COLLOCATION_POINTS = 4096
    NUM_RESIDUAL_SCOUT_POINTS = 500 # 用于侦察的点数，远少于训练点数

    # 1. 数据加载
    data_loader = DummyDataLoader(
        data_path=DATA_PATH,
        space_dims=SPACE_DIMS,
        num_samples=NUM_SAMPLES
    )
    training_data, test_data, dose_data = data_loader.get_training_data()

    # 2. 模型和采样器初始化
    pinn = PINNModel(
        dose_data=dose_data, 
        training_data=training_data,
        test_data=test_data,
        num_collocation_points=NUM_COLLOCATION_POINTS
    )
    kriging = GPUKriging()
    
    # 从 dose_data 中获取真实的物理边界
    world_min = dose_data['world_min']
    world_max = dose_data['world_max']
    
    sampler = AdaptiveSampler(world_min, world_max)
    
    # 初始配置点：在真实物理空间内采样
    current_collocation_points = (np.random.rand(NUM_COLLOCATION_POINTS, 3) * 
                                  (world_max - world_min) + world_min)

    # --- 3. 训练循环 ---
    for epoch in range(0, TOTAL_EPOCHS, ADAPTIVE_CYCLE_EPOCHS):
        
        print(f"\n--- 主循环周期: Epochs [{epoch} - {epoch + ADAPTIVE_CYCLE_EPOCHS - 1}] ---")

        # 2a. 使用当前的配置点，对PINN进行一轮常规训练
        print(f"PHASE 2a: 常规PINN训练...")
        pinn.run_training_cycle(
            epochs=ADAPTIVE_CYCLE_EPOCHS,
            collocation_points=current_collocation_points
        )
        
        print("\nPHASE 2b: 开始克里金引导的自适应采样...")
        
        # 2b. 残差"侦察"：用当前PINN计算一小批随机点的真实残差
        scout_points = (np.random.rand(NUM_RESIDUAL_SCOUT_POINTS, 3) *
                        (world_max - world_min) + world_min)
        true_residuals = pinn.compute_pde_residual(scout_points)
        
        # 2c. 克里金代理建模：训练Kriging模型来拟合残差分布
        kriging.fit(scout_points, true_residuals)

        # 2d. 自适应采样：使用训练好的Kriging模型生成下一批"更聪明"的配置点
        current_collocation_points = sampler.generate_new_collocation_points(
            kriging_model=kriging,
            num_points_to_sample=NUM_COLLOCATION_POINTS
        )
        print("PHASE 2b: ✅ 新的自适应配置点已生成。")

    print("\n" + "="*60)
    print("🎉 训练完成!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main() 