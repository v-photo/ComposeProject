import numpy as np
import sys
from pathlib import Path
import deepxde as dde
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- 解决WSL环境下的matplotlib显示问题 ---
import matplotlib
matplotlib.use('Agg')  # 使用无GUI的后端，避免Qt错误
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

# --- [全局] 实验控制开关 ---
ENABLE_KRIGING = True     # 🔧 控制是否启用克里金引导的重采样
ENABLE_DATA_INJECTION = False  # 🔧 控制是否启用数据注入策略
ENABLE_RAPID_IMPROVEMENT_EARLY_STOP = False  # 🔧 控制是否启用快速改善早停

# --- [全局] 探索率配置 ---
# 📊 探索率递减策略配置
# 计算公式: exploration_ratio = max(FINAL, INITIAL - (cycle-1) * DECAY_RATE)

# # 🎯 当前配置 (适中策略)
# INITIAL_EXPLORATION_RATIO = 0.20    # 初始探索率 (第1周期): 20%
# FINAL_EXPLORATION_RATIO = 0.05      # 最终探索率 (收敛值): 5%
# EXPLORATION_DECAY_RATE = 0.02       # 每周期递减率: 2%
# 👆 该配置下：第1周期20% → 第8周期5% → 之后保持5%

# 💡 其他常用配置示例 (取消注释使用):
# 
# 🚀 激进策略 (快速从探索转向利用)
# INITIAL_EXPLORATION_RATIO = 0.25    # 25%
# FINAL_EXPLORATION_RATIO = 0.02      # 2%
# EXPLORATION_DECAY_RATE = 0.05       # 5%
# # # 效果：第1周期25% → 第5周期5% → 第6周期2%
#
# 🐌 保守策略 (长期保持探索)
INITIAL_EXPLORATION_RATIO = 0.50    # 50%
FINAL_EXPLORATION_RATIO = 0.18      # 18%
EXPLORATION_DECAY_RATE = 0.3       # 4%
# # 效果：第1周期15% → 第8周期8% → 之后保持8%
#
# # 🎯 精准策略 (高利用率)
# INITIAL_EXPLORATION_RATIO = 0.30    # 30%
# FINAL_EXPLORATION_RATIO = 0.03      # 3%
# EXPLORATION_DECAY_RATE = 0.03       # 3%
# # 效果：第1周期30% → 第10周期3% → 之后保持3%

# --- [新增] 损失权重比值预设配置 ---
# 🔧 LOSS_RATIO 预设选项，方便快速切换测试不同比值对克里金效果的影响
# 
# 📊 克里金优化友好的推荐比值:
LOSS_RATIO_PHYSICS_DOMINANT_SUPER = 0.1   # 🔬 超级物理主导: 物理权重 > 数据权重 (10:1)
LOSS_RATIO_PHYSICS_DOMINANT = 0.5   # 🔬 物理主导: 物理权重 > 数据权重 (2:1)
LOSS_RATIO_PHYSICS_STRONG = 0.8     # ⚗️  物理优先: 物理约束稍强于数据拟合 (1.25:1)  
LOSS_RATIO_EQUAL_BALANCE = 1.0      # ⚖️  完全平衡: 物理权重 = 数据权重 (1:1)
LOSS_RATIO_CONSERVATIVE = 3.0       # 🛡️  保守策略: 强调物理约束，残差分布平滑
LOSS_RATIO_BALANCED = 5.0           # ⚖️  平衡策略: 兼顾数据拟合和物理约束 (推荐)
LOSS_RATIO_AGGRESSIVE = 8.0         # 🎯 激进策略: 偏重数据拟合，适合高质量数据
LOSS_RATIO_CURRENT_DEFAULT = 10.0   # 📌 当前默认: 可能对克里金效果不佳
LOSS_RATIO_TOO_HIGH = 15.0          # ⚠️  过高比值: 过度数据拟合，不利于克里金插值

# --- [新增] 动态损失权重策略配置 ---
# 🎯 当启用克里金重采样时，采用两阶段动态策略：
# 前一半训练：专注数据学习（高数据权重，无克里金重采样）
# 后一半训练：启用克里金重采样，逐渐降低数据权重
DYNAMIC_LOSS_RATIO_START = 10.0     # 🚀 动态策略起始比值：前3/4段训练的数据权重/物理权重比值
DYNAMIC_LOSS_RATIO_END = 0.1       # 🎯 动态策略结束比值：后1/4段训练结束时的比值
FIXED_LOSS_RATIO = 10              # 📌 固定策略比值：当不启用克里金时使用的固定比值

# 💡 使用建议:
# 🔬 物理主导策略 (loss_ratio < 1.0):
# - 优势: 强物理一致性，残差分布更平滑连续，有利于克里金空间插值
# - 适用: 物理方程高度可信、数据存在噪声、需要外推预测的场景
# - 风险: 可能在观测数据点附近拟合不够精确
#
# ⚖️ 平衡策略 (loss_ratio = 1.0 ~ 5.0):  
# - 优势: 兼顾数据拟合和物理约束，残差分布相对均匀
# - 适用: 大多数PINN应用，特别是有适量观测数据的情况
# - 推荐: 克里金优化的最佳起始选择
#
# 🎯 数据主导策略 (loss_ratio > 5.0):
# - 优势: 精确拟合观测数据，适合高质量密集数据
# - 适用: 数据质量很高、物理方程可能有近似误差的场景  
# - 风险: 残差在数据点附近被过度压制，不利于克里金插值
#
# 🧪 实验建议: 依次测试 [0.5, 1.0, 3.0, 5.0, 8.0, 10.0] 来找到最佳比值

# 四种实验模式:
# ENABLE_KRIGING=False, ENABLE_DATA_INJECTION=False: 仅周期性重启 (无自适应策略)
# ENABLE_KRIGING=False, ENABLE_DATA_INJECTION=True:  仅数据注入策略
# ENABLE_KRIGING=True,  ENABLE_DATA_INJECTION=False: 仅克里金重采样策略  
# ENABLE_KRIGING=True,  ENABLE_DATA_INJECTION=True:  完整自适应PINN

class DummyDataLoader:
    """
    一个数据加载器，用于从外部文件加载初始训练数据(替代原有的DummyDataLoader)。
    """
    def __init__(self, data_path: str, space_dims: np.ndarray, num_samples: int):
        self.data_path = data_path
        self.space_dims = space_dims
        self.num_samples = num_samples
        print(f"INFO: (DataLoader) Initialized with data_path='{self.data_path}'")

    def get_training_data(self, split_ratios: list = None, test_set_size: int = None):
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
            test_set_size (int, optional): 如果指定，将生成独立的测试集而非从训练数据分割。
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
        
        # [新增] 生成独立测试集（如果指定）
        if test_set_size is not None:
            print(f"INFO: (DataLoader) Generating independent test set of size {test_set_size}...")
            test_set = self._generate_independent_test_set(dose_data, test_set_size)
        else:
            test_set = None  # 将在下面的分割逻辑中处理
        
        # [新逻辑] 使用可配置的分割策略
        if split_ratios is None:
            # 默认行为：80/20 分割
            if test_set is None:
                main_train_set, test_set = train_test_split(all_sampled_data, test_size=0.2, random_state=42)
            else:
                main_train_set = all_sampled_data  # 全部用作训练数据
            reserve_pools = []
        else:
            if test_set is None and sum(split_ratios) >= 1.0:
                raise ValueError("split_ratios 的总和必须小于 1.0，以便为测试集留出空间。")

            remaining_data = all_sampled_data
            data_pools = []
            
            # 循环切分出主训练集和所有储备集
            current_total_fraction = 1.0
            for ratio in split_ratios:
                # 计算当前比例相对于剩余数据量的比例
                split_fraction = ratio / current_total_fraction
                
                # [修复] 数值稳定性检查，避免浮点数精度问题
                test_size_fraction = 1.0 - split_fraction
                if test_size_fraction < 1e-10:  # 如果非常接近0，则设为0
                    test_size_fraction = 0.0
                elif test_size_fraction > 1.0:  # 如果超过1，则设为1
                    test_size_fraction = 1.0
                
                # 如果test_size为0，说明剩余数据就是当前pool
                if test_size_fraction == 0.0:
                    pool = remaining_data
                    remaining_data = np.array([]).reshape(0, remaining_data.shape[1]) if len(remaining_data) > 0 else np.array([])
                else:
                    pool, remaining_data = train_test_split(remaining_data, test_size=test_size_fraction, random_state=42)
                
                data_pools.append(pool)
                current_total_fraction -= ratio

            main_train_set = data_pools[0]
            reserve_pools = data_pools[1:]
            
            # 如果没有独立测试集，则使用剩余数据
            if test_set is None:
                test_set = remaining_data
        
        print(f"INFO: (DataLoader) ✅ Split data into: Main training ({len(main_train_set)}), Test ({len(test_set)}), Reserve Pools ({len(reserve_pools)} pools).")
        if reserve_pools:
            for i, pool in enumerate(reserve_pools):
                print(f"    - Reserve Pool {i+1}: {len(pool)} points")

        return main_train_set, reserve_pools, test_set, dose_data

    def _generate_independent_test_set(self, dose_data: dict, test_set_size: int):
        """
        生成完全独立于训练数据的测试集，在整个物理域内均匀采样。
        
        Args:
            dose_data (dict): 包含物理域边界的数据字典
            test_set_size (int): 测试集大小
            
        Returns:
            np.ndarray: 测试集数据 [x, y, z, value]
        """
        # 使用 DataLoader.sample_training_points 在整个域内采样测试点
        test_points, test_values, _ = DataLoader.sample_training_points(
            dose_data, 
            num_samples=test_set_size,
            sampling_strategy='uniform',  # 使用均匀采样
        )
        
        # 合并为 [x, y, z, value] 格式
        test_set = np.hstack([test_points, test_values.reshape(-1, 1)])
        print(f"INFO: (DataLoader) ✅ Generated independent test set with {len(test_set)} points.")
        return test_set

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
    def __init__(self, dose_data: dict, training_data: np.ndarray, test_data: np.ndarray, num_collocation_points: int, network_layers=[3, 64, 64, 64, 1], lr=1e-3, loss_ratio=10.0):
        """
        初始化PINN模型，但与PINNTrainer不同，这里只做准备工作，不开始训练。
        
        Args:
            dose_data (dict): 从DataLoader加载的数据字典。
            training_data (np.ndarray): 稀疏训练数据 [x,y,z,value]。
            test_data (np.ndarray): 稀疏测试数据 [x,y,z,value]。
            num_collocation_points (int): 求解域点的数量。
            network_layers (list): 神经网络结构。
            lr (float): 学习率。
            loss_ratio (float): 数据损失权重与物理损失权重的比值 (数据损失权重 / 物理损失权重)，默认10.0。
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
        self.loss_ratio = loss_ratio # [新增] 保存损失比值以备重编译时使用
        
        # [新增] MRE历史记录列表
        self.mre_history = []
        self.epoch_history = []
        
        # 8. 编译模型，加入自定义指标
        self.compile_model()
        print(f"INFO: (PINNModel) ✅ Model compiled with loss_ratio={loss_ratio:.1f} (data/physics weight ratio).")
        print(f"      └─ Loss weights: [physics={1.0:.1f}, data={loss_ratio:.1f}]")
        
    def compile_model(self):
        """将模型编译封装成一个方法，方便重用。"""
        # [修正] 在这里设置指标函数的显示名称
        # self.mean_relative_error_metric.__name__ = "MRE_test_set" # [修正] 移除此行，不能为类方法设置__name__

        # [新增] 基于loss_ratio动态计算loss_weights
        # loss_ratio = 数据损失权重 / 物理损失权重
        physics_weight = 1.0
        data_weight = self.loss_ratio
        loss_weights = [physics_weight, data_weight]

        self.model.compile(
            "adam", 
            lr=self.lr, 
            loss_weights=loss_weights, 
            external_trainable_variables=[self.log_k_pinn],
            metrics=[self.mean_relative_error_metric] # [修正] 传递函数对象，而不是字符串
        )
    
    def update_loss_ratio(self, new_loss_ratio: float):
        """
        [新增] 动态更新损失权重比值并重新编译模型。
        
        Args:
            new_loss_ratio (float): 新的数据损失权重/物理损失权重比值
        """
        if abs(self.loss_ratio - new_loss_ratio) > 1e-6:  # 避免不必要的重编译
            old_ratio = self.loss_ratio
            self.loss_ratio = new_loss_ratio
            self.compile_model()
            print(f"INFO: (PINNModel) 损失权重比值已更新: {old_ratio:.2f} → {new_loss_ratio:.2f}")
            print(f"      └─ 新的损失权重: [物理={1.0:.1f}, 数据={new_loss_ratio:.1f}]")
            
            # [修正] 重新编译后需要重建训练状态
            self.model.train(iterations=0, display_every=100000)
        else:
            print(f"INFO: (PINNModel) 损失权重比值未变化，跳过重编译 (当前: {self.loss_ratio:.2f})")

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

            # 条件2: 快速提升 (Rapid Improvement) - 🔧 可选择是否启用
            if ENABLE_RAPID_IMPROVEMENT_EARLY_STOP and stopper.should_stop:
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
        cycle_number: int = 1
    ) -> tuple[np.ndarray, float]:
        """
        使用Kriging模型引导生成新的配置点。
        Args:
            kriging_model: 训练好的残差代理模型。
            num_points_to_sample: 需要生成的总点数。
            cycle_number: 当前是第几个自适应周期，用于动态调整探索策略。
        Returns:
            tuple: (新的配置点集, 使用的探索率)
        """
        # [新逻辑] 基于周期数和全局配置计算探索率
        exploration_ratio = max(
            FINAL_EXPLORATION_RATIO,
            INITIAL_EXPLORATION_RATIO - (cycle_number - 1) * EXPLORATION_DECAY_RATE
        )
        
        print(f"INFO: (AdaptiveSampler) 周期性克里金重采样 (第{cycle_number}次)")
        print(f"      探索率: {exploration_ratio:.1%} (初始:{INITIAL_EXPLORATION_RATIO:.1%} → 最终:{FINAL_EXPLORATION_RATIO:.1%})")

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
        
        return np.vstack([exploitation_points, exploration_points]), exploration_ratio

def main():
    """
    主函数，编排整个"克里金引导的自适应PINN训练"流程。
    """
    # --- 1. 初始化 ---
    # !! 注意: DOMAIN_BOUNDS 现在仅用于可视化或采样器，实际物理边界由加载的数据决定 !!
    DOMAIN_BOUNDS = np.array([[0., 0., 0.], [1., 1., 1.]]) 
    TOTAL_EPOCHS = 8000
    ADAPTIVE_CYCLE_EPOCHS = 2000  # 每多少个epoch执行一次自适应调整
    DETECT_EPOCHS = 500 # 每100轮检测一次性能 [修正注释]
    DATA_SPLIT_RATIOS = [0.5] + [0.1]*5

    # --- [新增] 动态损失权重策略 ---
    # 🎯 根据是否启用克里金重采样选择不同的损失权重策略
    if ENABLE_KRIGING:
        # 启用克里金时，使用两阶段动态策略
        INITIAL_LOSS_RATIO = DYNAMIC_LOSS_RATIO_START  # 前半段：高数据权重，专注学习数据
        FINAL_LOSS_RATIO = DYNAMIC_LOSS_RATIO_END      # 后半段：降低数据权重，利于克里金
        USE_DYNAMIC_STRATEGY = True
        strategy_desc = f"动态策略: {INITIAL_LOSS_RATIO:.1f} → {FINAL_LOSS_RATIO:.1f}"
    else:
        # 不启用克里金时，使用固定损失权重
        INITIAL_LOSS_RATIO = FIXED_LOSS_RATIO
        FINAL_LOSS_RATIO = FIXED_LOSS_RATIO  
        USE_DYNAMIC_STRATEGY = False
        strategy_desc = f"固定策略: {FIXED_LOSS_RATIO:.1f}"

    print("\n" + "="*60)
    print("🚀 开始执行：自适应PINN训练实验")
    print("="*60)
    print(f"📋 实验配置:")
    print(f"   - 损失权重策略: {strategy_desc}")
    if USE_DYNAMIC_STRATEGY:
        print(f"     ├─ 前3/4段训练: 数据权重/物理权重 = {INITIAL_LOSS_RATIO:.1f} (专注数据学习)")
        print(f"     ├─ 后1/4段训练: {INITIAL_LOSS_RATIO:.1f} → {FINAL_LOSS_RATIO:.1f} (指数衰减，物理优化)")
        print(f"     ├─ 衰减策略: 快速降低数据权重，让物理约束充分发挥作用")
        print(f"     └─ 克里金策略: 仅后1/4段启用，利用物理主导下的平滑残差分布")
    else:
        print(f"     └─ 固定损失权重: [物理={1.0:.1f}, 数据={INITIAL_LOSS_RATIO:.1f}]")
    print(f"   - 克里金引导采样: {'✅ 启用' if ENABLE_KRIGING else '❌ 禁用'}")
    if ENABLE_KRIGING:
        print(f"     └─ 探索率策略: {INITIAL_EXPLORATION_RATIO:.1%} → {FINAL_EXPLORATION_RATIO:.1%} (每周期-{EXPLORATION_DECAY_RATE:.1%})")
    print(f"   - 数据注入策略: {'✅ 启用' if ENABLE_DATA_INJECTION else '❌ 禁用'}")
    print(f"   - 快速改善早停: {'✅ 启用' if ENABLE_RAPID_IMPROVEMENT_EARLY_STOP else '❌ 禁用'}")
    print(f"   - 总训练轮数: {TOTAL_EPOCHS}")
    print(f"   - 干预周期: 每 {ADAPTIVE_CYCLE_EPOCHS} 轮")
    
    # 确定实验类型
    if ENABLE_KRIGING and ENABLE_DATA_INJECTION:
        exp_type = "完整自适应PINN (数据注入 + 克里金重采样)"
    elif ENABLE_KRIGING and not ENABLE_DATA_INJECTION:
        exp_type = "仅克里金重采样策略"
    elif not ENABLE_KRIGING and ENABLE_DATA_INJECTION:
        exp_type = "仅数据注入策略"
    else:
        exp_type = "仅周期性重启 (无自适应策略)"
    
    print(f"   - 实验类型: {exp_type}")
    print("="*60 + "\n")
    
    # --- 数据加载参数 ---
    DATA_PATH = "PINN/DATA.xlsx"
    SPACE_DIMS = np.array([20.0, 10.0, 10.0])
    NUM_SAMPLES = 100
    
    # --- 模型训练参数 ---
    NUM_COLLOCATION_POINTS = 4096
    NUM_RESIDUAL_SCOUT_POINTS = 5000 # 用于侦察的点数，远少于训练配置点数

    # 1. 数据加载
    data_loader = DummyDataLoader(
        data_path=DATA_PATH,
        space_dims=SPACE_DIMS,
        num_samples=NUM_SAMPLES
    )
    main_train_set, reserve_data_pools, test_data, dose_data = data_loader.get_training_data(
        split_ratios=DATA_SPLIT_RATIOS,
        test_set_size=300  # [新增] 独立测试集大小
    )

    # 2. 模型和采样器初始化
    world_min = dose_data['world_min']
    world_max = dose_data['world_max']
    
    # 使用初始损失权重比值初始化PINN模型
    pinn = PINNModel(
        dose_data=dose_data, 
        training_data=main_train_set, # [新] 只用主训练集初始化
        test_data=test_data,
        num_collocation_points=NUM_COLLOCATION_POINTS,
        loss_ratio=INITIAL_LOSS_RATIO  # [修改] 使用动态策略的初始比值
    )
    kriging = GPUKriging()
    sampler = AdaptiveSampler(domain_bounds=np.vstack([world_min, world_max]))
    
    # 初始配置点：在真实物理空间内采样
    current_collocation_points = (np.random.rand(NUM_COLLOCATION_POINTS, 3) * 
                                  (world_max - world_min) + world_min)

    # --- 3. [修正] 训练循环 ---
    # 使用 while 循环来确保总训练轮数达标
    total_epochs_trained = 0
    cycle_counter = 0  # 🔧 新增：周期计数器，用于可靠的克里金触发
    
    # [新增] 重要事件记录列表，用于图表标注
    important_events = []  # 格式: [(epoch, event_type, description), ...]
    
    # [新增] 动态策略相关变量
    training_phase_transition_point = TOTAL_EPOCHS * 3 // 4  # 训练阶段转换点（3/4处）
    kriging_enabled_this_cycle = False  # 当前周期是否启用克里金
    current_loss_ratio = INITIAL_LOSS_RATIO  # 当前使用的损失权重比值
    
    print(f"\n🎯 动态训练策略:")
    if USE_DYNAMIC_STRATEGY:
        print(f"   📚 第一阶段 (0-{training_phase_transition_point}轮): 数据学习期")
        print(f"      ├─ 损失权重比值: {INITIAL_LOSS_RATIO:.1f} (数据主导，充分学习观测数据)")
        print(f"      └─ 克里金重采样: ❌ 禁用 (高数据权重导致残差分布不均，不利于克里金)")
        print(f"   🔬 第二阶段 ({training_phase_transition_point}-{TOTAL_EPOCHS}轮): 物理约束优化期")
        print(f"      ├─ 损失权重比值: {INITIAL_LOSS_RATIO:.1f} → {FINAL_LOSS_RATIO:.1f} (指数衰减)")
        print(f"      ├─ 衰减特点: 快速降低数据权重，强化物理约束")
        print(f"      └─ 克里金重采样: ✅ 启用 (物理主导产生平滑残差分布，利于克里金)")
    else:
        print(f"   📊 固定策略: 损失权重比值保持 {INITIAL_LOSS_RATIO:.1f}")
    print()

    while total_epochs_trained < TOTAL_EPOCHS:
        remaining_total_epochs = TOTAL_EPOCHS - total_epochs_trained
        
        # 本次自适应周期的最大训练轮数，不能超过总剩余轮数
        cycle_max_epochs = min(ADAPTIVE_CYCLE_EPOCHS, remaining_total_epochs)
        
        print(f"\n--- 主循环周期: 目标训练 {total_epochs_trained} -> {total_epochs_trained + cycle_max_epochs} ---")
        
        # [新增] 动态策略: 检查是否需要更新损失权重或克里金状态
        if USE_DYNAMIC_STRATEGY:
            # 计算当前训练进度比例
            current_progress = total_epochs_trained / TOTAL_EPOCHS
            
            # 检查是否已经进入第二阶段（克里金启用阶段）
            is_in_kriging_phase = total_epochs_trained >= training_phase_transition_point
            
            if is_in_kriging_phase:
                # 第二阶段：启用克里金（物理主导产生平滑残差分布，利于克里金）
                kriging_enabled_this_cycle = True
            else:
                # 第一阶段：禁用克里金（数据权重高导致残差分布不均匀，不利于克里金）
                kriging_enabled_this_cycle = False
            
            if is_in_kriging_phase:
                # 第二阶段：指数衰减损失权重比值（从数据主导转向物理主导）
                progress_in_second_phase = (total_epochs_trained - training_phase_transition_point) / (TOTAL_EPOCHS - training_phase_transition_point)
                
                # 使用指数衰减策略，快速降低数据权重，让物理约束发挥作用
                # 公式：ratio = start * ((end/start)^(progress^2))
                # progress^2 让前期快速下降，确保物理权重主导期有足够训练时间
                decay_factor = (FINAL_LOSS_RATIO / INITIAL_LOSS_RATIO) ** (progress_in_second_phase ** 2)
                new_loss_ratio = INITIAL_LOSS_RATIO * decay_factor
                
                # 确保不低于最终目标值
                new_loss_ratio = max(new_loss_ratio, FINAL_LOSS_RATIO)
                
                # 阶段转换提示（仅在刚进入第二阶段时显示）
                if total_epochs_trained == training_phase_transition_point or (total_epochs_trained < training_phase_transition_point + ADAPTIVE_CYCLE_EPOCHS and cycle_counter <= 1):
                    print(f"\n🔄 进入第二阶段: 物理约束优化期")
                    print(f"   📉 开始降低数据权重，强化物理约束")
                    print(f"   ✅ 克里金重采样已启用（利用物理主导下的平滑残差分布）")
                    important_events.append((
                        total_epochs_trained, 
                        'phase_transition', 
                        '进入物理约束期，启用克里金'
                    ))
            else:
                # 第一阶段：固定数据主导权重
                new_loss_ratio = INITIAL_LOSS_RATIO
            
            # 更新损失权重（如果有变化）
            if abs(current_loss_ratio - new_loss_ratio) > 1e-6:
                pinn.update_loss_ratio(new_loss_ratio)
                current_loss_ratio = new_loss_ratio
                
                # 记录损失权重变化事件
                important_events.append((
                    total_epochs_trained, 
                    'loss_ratio_update', 
                    f'损失权重比值更新至 {new_loss_ratio:.2f}'
                ))
            
            print(f"📊 当前训练状态:")
            print(f"   - 训练进度: {current_progress:.1%} ({total_epochs_trained}/{TOTAL_EPOCHS})")
            print(f"   - 训练阶段: {'第二阶段(物理约束优化)' if is_in_kriging_phase else '第一阶段(数据学习)'}")
            print(f"   - 损失权重比值: {current_loss_ratio:.2f}")
            if is_in_kriging_phase:
                phase_progress = (total_epochs_trained - training_phase_transition_point) / (TOTAL_EPOCHS - training_phase_transition_point)
                print(f"   - 第二阶段进度: {phase_progress:.1%}")
                if current_loss_ratio < 1.0:
                    print(f"   - 💡 当前物理权重主导 (比值 < 1.0)，有利于克里金优化")
                elif current_loss_ratio < 3.0:
                    print(f"   - 📊 当前接近平衡状态，正在向物理主导转变")
                else:
                    print(f"   - 📈 当前数据权重仍占主导，正在逐步降低")
            print(f"   - 克里金重采样: {'✅ 启用' if kriging_enabled_this_cycle else '❌ 禁用'}")
        else:
            # 固定策略：克里金状态由全局开关决定
            kriging_enabled_this_cycle = ENABLE_KRIGING

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
        cycle_counter += 1  # 🔧 增加周期计数

        print(f"\nINFO: 本周期实际训练 {epochs_this_cycle} 轮. 总训练进度: {total_epochs_trained}/{TOTAL_EPOCHS}")
        print(f"🔢 周期计数: 第 {cycle_counter} 个周期完成")
        
        # 🔍 新增：性能分析 - 记录当前周期的最终MRE
        current_mre = pinn.model.train_state.metrics_test[-1] if pinn.model.train_state.metrics_test else float('inf')
        print(f"📊 周期性能: 第{cycle_counter}周期结束时MRE = {current_mre:.6f} (训练{epochs_this_cycle}轮)")
        
        # 🔍 如果是第2+周期，计算改善率
        if cycle_counter > 1 and hasattr(main, 'previous_cycle_mre'):
            improvement = main.previous_cycle_mre - current_mre
            improvement_rate = improvement / main.previous_cycle_mre if main.previous_cycle_mre > 0 else 0
            print(f"    └─ 相比上周期改善: {improvement:.6f} ({improvement_rate:.2%})")
            
            # 评估收敛速度
            if improvement_rate > 0.1:
                print(f"    🚀 快速收敛! 改善率 > 10%")
            elif improvement_rate > 0.05:
                print(f"    📈 良好收敛! 改善率 > 5%")
            elif improvement_rate > 0:
                print(f"    📊 缓慢改善")
            else:
                print(f"    ⚠️  性能下降或停滞")
        
        # 保存当前MRE供下一周期比较
        if not hasattr(main, 'previous_cycle_mre'):
            main.previous_cycle_mre = current_mre
        else:
            main.previous_cycle_mre = current_mre

        # 如果已经训练够了，就提前结束主循环
        if total_epochs_trained >= TOTAL_EPOCHS:
            print("\nINFO: 总训练轮数已达到目标，结束自适应训练。")
            break

        # 2. 🔧 改用周期计数器进行可靠的周期性干预触发
        should_trigger_intervention = cycle_counter > 0  # 每个周期都检查是否需要干预
        print(f"🔍 干预触发检查: 周期 {cycle_counter} 完成，应该触发干预 → {should_trigger_intervention}")
        
        if should_trigger_intervention:
            print("\n" + "!"*60)
            
            # 根据启用的策略确定干预类型描述
            intervention_types = []
            if ENABLE_DATA_INJECTION:
                intervention_types.append("数据注入")
            if ENABLE_KRIGING:
                intervention_types.append("克里金重采样")
            
            if intervention_types:
                intervention_desc = " + ".join(intervention_types)
                print(f"!! 训练达到 {total_epochs_trained} 轮，触发周期性干预: {intervention_desc} !!")
            else:
                print(f"!! 训练达到 {total_epochs_trained} 轮，触发周期性重启 (无自适应策略) !!")
            print("!"*60)

            # --- 干预措施 1: 注入新数据 (如果启用且还有数据) ---
            if ENABLE_DATA_INJECTION:
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
                        f'周期性数据注入 (+{len(data_to_inject)}点, 第{cycle_counter}次)'
                    ))
                else:
                    print("\nWARNING: 数据注入已启用，但已无更多储备数据可注入。")
            else:
                print("\nPHASE A: 数据注入已禁用，跳过此阶段。")

            # --- 干预措施 2: 克里金引导的自适应采样 (动态策略控制) ---
            if kriging_enabled_this_cycle:  # [修改] 使用动态策略控制的克里金启用状态
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
                
                # 🔍 残差质量分析
                high_residual_ratio = np.mean(true_residuals > np.mean(true_residuals) * 2)
                print(f"      - 高残差点比例: {high_residual_ratio:.1%} (残差>2倍均值)")
                
                # 🔍 新增：残差分布质量诊断 (用于评估loss_ratio效果)
                residual_std = np.std(true_residuals)
                residual_cv = residual_std / (np.mean(true_residuals) + 1e-10)  # 变异系数
                print(f"      - 残差变异系数: {residual_cv:.3f} (标准差/均值，越小越利于克里金)")
                print(f"      - 当前loss_ratio: {current_loss_ratio:.1f} 的残差分布特征分析完成")
                
                # 💡 残差质量评价
                if residual_cv < 0.5:
                    print(f"      - ✅ 残差分布较为均匀，有利于克里金插值")
                elif residual_cv < 1.0:
                    print(f"      - ⚠️  残差分布中等变异，克里金效果可能一般")
                else:
                    print(f"      - ❌ 残差分布变异过大，建议进一步降低loss_ratio改善物理约束")
                
                # 克里金代理建模
                print("    🔧 开始训练克里金代理模型...")
                kriging.fit(scout_points, true_residuals)

                # 自适应采样
                kriging_epoch = pinn.model.train_state.step or 0
                num_collocation_to_generate = pinn.data.num_domain
                print(f"INFO: Dynamically calculated {num_collocation_to_generate} collocation points to generate.")

                # 🔧 使用周期计数器作为周期编号
                current_collocation_points, used_exploration_ratio = sampler.generate_new_collocation_points(
                    kriging_model=kriging,
                    num_points_to_sample=num_collocation_to_generate,
                    cycle_number=cycle_counter
                )
                print("PHASE B: ✅ 新的自适应配置点已生成。")
                
                # 🔍 新增：评估新配置点的预期残差质量
                predicted_residuals_new = kriging.predict(current_collocation_points)
                old_residuals_sample = pinn.compute_pde_residual(current_collocation_points[:100])  # 采样100个点评估
                print(f"    📊 新配置点质量评估:")
                print(f"      - 克里金预测残差: Mean={np.mean(predicted_residuals_new):.4e}, Max={np.max(predicted_residuals_new):.4e}")
                print(f"      - 实际残差(采样): Mean={np.mean(old_residuals_sample):.4e}, Max={np.max(old_residuals_sample):.4e}")
                residual_prediction_accuracy = np.corrcoef(
                    predicted_residuals_new[:100], old_residuals_sample
                )[0,1] if len(old_residuals_sample) == 100 else 0
                print(f"      - 克里金预测准确度: {residual_prediction_accuracy:.3f} (相关系数)")
                
                # [新增] 记录周期性克里金应用事件
                important_events.append((
                    kriging_epoch, 
                    'kriging_resampling', 
                    f'周期性克里金重采样 (第{cycle_counter}次, 探索率:{used_exploration_ratio:.1%})'
                ))
            else:
                # [修改] 根据动态策略状态给出不同的提示信息
                if USE_DYNAMIC_STRATEGY and total_epochs_trained < training_phase_transition_point:
                    print("\nPHASE B: 克里金重采样在第一阶段禁用，专注数据学习。")
                    print("INFO: 配置点保持不变，当前处于数据学习期。")
                elif not ENABLE_KRIGING:
                    print("\nPHASE B: 克里金重采样已在全局配置中禁用，跳过此阶段。")
                    print("INFO: 配置点保持不变，仅依靠数据注入策略。")
                else:
                    print("\nPHASE B: 克里金重采样本周期未启用。")
                    print("INFO: 配置点保持不变。")

            # --- 周期性干预完成 ---
            # 确定干预类型描述（考虑动态策略）
            intervention_parts = []
            if ENABLE_DATA_INJECTION:
                intervention_parts.append("数据注入")
            if kriging_enabled_this_cycle:
                intervention_parts.append("克里金重采样")
            
            if intervention_parts:
                intervention_desc = " + ".join(intervention_parts)
                if USE_DYNAMIC_STRATEGY:
                    phase_desc = "第二阶段" if total_epochs_trained >= training_phase_transition_point else "第一阶段"
                    intervention_desc += f" ({phase_desc})"
            else:
                intervention_desc = "周期性重启"
                if USE_DYNAMIC_STRATEGY:
                    intervention_desc += " (第一阶段)"
            
            print(f"\nINFO: 第 {cycle_counter} 次周期性干预完成 ({intervention_desc})。")
            print("-" * 60)
        
        else:
            # 🔧 这个分支现在应该不会被执行，因为每个周期都会触发干预检查
            print(f"\n⚠️  未预期的情况：周期 {cycle_counter} 不应该跳过干预检查！")
            pass

    print("\n" + "="*60)
    print("🎉 训练完成!")
    print("="*60 + "\n")

    # --- 4. 训练原始PINN作为对比模型 ---
    print("\n" + "="*60)
    print("🚀 开始训练原始PINN作为对比基线")
    print("="*60 + "\n")

    # [修正] 为基线模型准备与自适应PINN相同的训练数据
    # 获取自适应PINN实际使用的训练数据，确保公平对比
    adaptive_training_data = pinn.data.bcs[0].points  # 自适应PINN的实际训练点坐标
    adaptive_training_values = pinn.data.bcs[0].values.cpu().numpy()  # 对应的值（对数尺度）
    
    # 将对数尺度的值转换回线性尺度，然后合并为 [x,y,z,value] 格式
    adaptive_training_linear_values = np.exp(adaptive_training_values)
    full_training_data = np.hstack([adaptive_training_data, adaptive_training_linear_values])
    
    print(f"INFO: 基线PINN设置:")
    print(f"   - 训练点数: {len(full_training_data)} (与自适应PINN最终使用的训练点相同)")
    print(f"   - 损失权重策略: 固定策略 (loss_ratio = {FIXED_LOSS_RATIO:.1f})")
    print(f"   - 配置点采样: 固定随机采样")
    print(f"   - 对比目的: 验证动态策略相对于固定策略的优势")

    pinn_baseline = PINNModel(
        dose_data=dose_data, 
        training_data=full_training_data, # [新] 使用全部训练数据
        test_data=test_data,
        num_collocation_points=NUM_COLLOCATION_POINTS,
        loss_ratio=FIXED_LOSS_RATIO  # [修正] 基线PINN使用固定策略确保公平对比
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
    
    # [新增] 获取两个模型的训练点数
    adaptive_train_points = pinn.data.bcs[0].points.shape[0]
    baseline_train_points = pinn_baseline.data.bcs[0].points.shape[0]

    # 动态模型名称
    if ENABLE_KRIGING and ENABLE_DATA_INJECTION:
        adaptive_model_name = "完整自适应PINN"
    elif ENABLE_KRIGING and not ENABLE_DATA_INJECTION:
        adaptive_model_name = "仅克里金重采样PINN"
    elif not ENABLE_KRIGING and ENABLE_DATA_INJECTION:
        adaptive_model_name = "仅数据注入PINN"
    else:
        adaptive_model_name = "周期性重启PINN"
    
    print(f"\n{'模型':<32} | {'平均相对误差 (MRE)':<20} | {'训练点数':<12}")
    print("-" * 74)
    print(f"{adaptive_model_name:<32} | {mre_adaptive:<20.6%} | {adaptive_train_points:<12d}")
    print(f"{'原始PINN (固定采样)':<32} | {mre_baseline:<20.6%} | {baseline_train_points:<12d}")
    print("-" * 74)
    
    # [新增] 训练效率对比
    print(f"\n📊 训练效率分析:")
    print(f"   自适应PINN使用了 {adaptive_train_points} 个训练点，达到 MRE = {mre_adaptive:.6%}")
    print(f"   基线PINN使用了 {baseline_train_points} 个训练点，达到 MRE = {mre_baseline:.6%}")
    if adaptive_train_points != baseline_train_points:
        efficiency_ratio = baseline_train_points / adaptive_train_points
        print(f"   训练点效率比: {efficiency_ratio:.2f}x (自适应PINN vs 基线PINN)")
    print(f"   独立测试集大小: {len(test_data)} 点")
    
    # 🔍 新增：收敛效率分析
    print(f"\n⚡ 收敛效率对比:")
    if mre_adaptive < mre_baseline:
        improvement = (mre_baseline - mre_adaptive) / mre_baseline
        print(f"   🎯 自适应PINN表现更优: 相对改善 {improvement:.2%}")
        print(f"   💡 克里金引导策略有效!")
    elif mre_adaptive > mre_baseline:
        degradation = (mre_adaptive - mre_baseline) / mre_baseline  
        print(f"   ⚠️  自适应PINN表现略差: 相对下降 {degradation:.2%}")
        print(f"   🔧 建议调整探索率策略或增加训练轮数")
    else:
        print(f"   📊 两种方法性能相当")
    
    # 计算收敛速度指标
    adaptive_epochs_to_convergence = len(pinn.mre_history)
    baseline_epochs_to_convergence = len(pinn_baseline.mre_history)
    
    print(f"\n🏃‍♂️ 收敛速度分析:")
    print(f"   自适应PINN: {adaptive_epochs_to_convergence} 次评估到达 MRE={mre_adaptive:.6%}")
    print(f"   基线PINN: {baseline_epochs_to_convergence} 次评估到达 MRE={mre_baseline:.6%}")
    
    if adaptive_epochs_to_convergence < baseline_epochs_to_convergence:
        speed_improvement = (baseline_epochs_to_convergence - adaptive_epochs_to_convergence) / baseline_epochs_to_convergence
        print(f"   🚀 自适应PINN收敛更快: 减少 {speed_improvement:.1%} 的评估次数")
    else:
        print(f"   📊 收敛速度相当或需要更多评估")

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
        # 使用与前面一致的模型名称
        if ENABLE_KRIGING and ENABLE_DATA_INJECTION:
            adaptive_label = "完整自适应PINN"
        elif ENABLE_KRIGING and not ENABLE_DATA_INJECTION:
            adaptive_label = "仅克里金重采样PINN"
        elif not ENABLE_KRIGING and ENABLE_DATA_INJECTION:
            adaptive_label = "仅数据注入PINN"
        else:
            adaptive_label = "周期性重启PINN"
            
        ax.plot(pinn.epoch_history, pinn.mre_history, 
                label=adaptive_label, linewidth=2, alpha=0.8, color='blue')
    
    # 绘制基线PINN的MRE历史
    if pinn_baseline.epoch_history and pinn_baseline.mre_history:
        ax.plot(pinn_baseline.epoch_history, pinn_baseline.mre_history, 
                label='原始PINN (固定采样)', linewidth=2, alpha=0.8, color='red')
    
    # 添加重要事件标注
    if important_events:
        print(f"INFO: 标注 {len(important_events)} 个重要时间点...")
        
        # 定义事件类型的颜色和样式
        event_styles = {
            'data_injection': {'color': 'green', 'linestyle': '--', 'alpha': 0.7, 'label': '数据注入'},
            'kriging_resampling': {'color': 'orange', 'linestyle': '-.', 'alpha': 0.7, 'label': '克里金重采样'},
            'phase_transition': {'color': 'purple', 'linestyle': ':', 'alpha': 0.7, 'label': '阶段转换'},
            'loss_ratio_update': {'color': 'red', 'linestyle': '-', 'alpha': 0.5, 'label': '权重更新'}
        }
        
        # 智能标注位置算法：避免重叠
        def get_smart_annotation_positions(events, y_min, y_max):
            """智能计算标注位置，避免重叠"""
            if not events:
                return []
            
            positions = []
            sorted_events = sorted(events, key=lambda x: x[0])  # 按epoch排序
            
            # 动态确定层级数量和位置
            num_events = len(sorted_events)
            max_levels = min(6, max(3, num_events))  # 至少3层，最多6层
            
            # 定义可用的y位置层级（从上到下，对数空间均匀分布）
            y_levels = []
            if y_max > y_min and y_max / y_min > 10:  # 对数尺度
                log_min, log_max = np.log10(y_min), np.log10(y_max)
                log_positions = np.linspace(log_max * 0.95, log_max * 0.5, max_levels)
                y_levels = [10 ** pos for pos in log_positions]
            else:  # 线性尺度
                y_levels = np.linspace(y_max * 0.95, y_max * 0.5, max_levels)
            
            # 计算最小距离（基于epoch范围的自适应）
            if len(sorted_events) > 1:
                epoch_range = sorted_events[-1][0] - sorted_events[0][0]
                min_distance = max(200, epoch_range * 0.05)  # 至少200epoch或总范围的5%
            else:
                min_distance = 200
            
            for i, (epoch, event_type, description) in enumerate(sorted_events):
                best_level = 0
                min_conflicts = float('inf')
                
                # 尝试每个层级，选择冲突最少的
                for level_idx in range(len(y_levels)):
                    conflicts = 0
                    for prev_epoch, prev_level in positions:
                        if (abs(epoch - prev_epoch) < min_distance and 
                            level_idx == prev_level):
                            conflicts += 1
                    
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_level = level_idx
                        if conflicts == 0:  # 找到无冲突层级，立即使用
                            break
                
                positions.append((epoch, best_level))
            
            return [(epoch, y_levels[level]) for epoch, level in positions]
        
        # 获取当前y轴范围
        y_min, y_max = ax.get_ylim()
        
        # 计算智能标注位置
        annotation_positions = get_smart_annotation_positions(important_events, y_min, y_max)
        
        # 绘制标注
        legend_handles = {}
        for i, ((epoch, event_type, description), (_, y_pos)) in enumerate(zip(important_events, annotation_positions)):
            style = event_styles.get(event_type, {'color': 'gray', 'linestyle': '-', 'alpha': 0.5, 'label': '其他'})
            
            # 绘制垂直线（从底部到顶部）
            line = ax.axvline(x=epoch, color=style['color'], linestyle=style['linestyle'], 
                             alpha=style['alpha'], linewidth=1.5)
            
            # 收集图例元素（避免重复）
            if event_type not in legend_handles:
                legend_handles[event_type] = line
            
            # 缩短描述文本以避免过长
            short_description = description.split('(')[0].strip()  # 只取括号前的部分
            if len(short_description) > 15:
                short_description = short_description[:12] + "..."
            
            # 计算连接线的起始点（垂直线顶部附近）
            line_start_y = y_max * 0.98
            
            # 绘制连接线（从垂直线顶部到标注框）
            if abs(y_pos - line_start_y) > y_max * 0.02:  # 只有当标注不在顶部时才画连接线
                ax.annotate('', xy=(epoch, y_pos), xytext=(epoch, line_start_y),
                           arrowprops=dict(arrowstyle='-', color=style['color'], 
                                         alpha=0.4, lw=1, linestyle='dotted'))
            
            # 添加文本标注（使用智能位置）
            annotation = ax.annotate(
                f'{short_description}\n(E{epoch})',
                xy=(epoch, y_pos),
                xytext=(8, 0),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=style['color'], 
                         alpha=0.15, edgecolor=style['color'], linewidth=0.5),
                fontsize=8,
                ha='left',
                va='center',
                weight='normal'
            )
            
            # 添加一个小圆点标记在标注位置
            ax.plot(epoch, y_pos, 'o', color=style['color'], markersize=4, 
                   alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
        
        # 添加图例说明（只显示实际出现的事件类型）
        if legend_handles:
            from matplotlib.lines import Line2D
            legend_elements = []
            for event_type, line in legend_handles.items():
                style = event_styles.get(event_type, {'color': 'gray', 'linestyle': '-', 'label': '其他'})
                legend_elements.append(
                    Line2D([0], [0], color=style['color'], linestyle=style['linestyle'], 
                          label=style['label'])
                )
            
            # 创建第二个图例
            second_legend = ax.legend(handles=legend_elements, loc='upper right', 
                                     fontsize=9, title='重要事件', title_fontsize=10,
                                     framealpha=0.9)
            ax.add_artist(second_legend)  # 保持原有图例
    
    # 设置图表属性
    ax.set_xlabel('训练轮数 (Epochs)', fontsize=12)
    ax.set_ylabel('平均相对误差 (MRE)', fontsize=12)
    
    # 动态图表标题
    if ENABLE_KRIGING and ENABLE_DATA_INJECTION:
        title_suffix = "完整自适应"
    elif ENABLE_KRIGING and not ENABLE_DATA_INJECTION:
        title_suffix = "仅克里金重采样"
    elif not ENABLE_KRIGING and ENABLE_DATA_INJECTION:
        title_suffix = "仅数据注入"
    else:
        title_suffix = "周期性重启"
    
    ax.set_title(f'{title_suffix}PINN vs 基线PINN: 训练过程MRE对比', fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=11)  # 调整原图例位置
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # 使用对数坐标更好地显示误差变化
    
    # 保存图表
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # 动态文件名
    if ENABLE_KRIGING and ENABLE_DATA_INJECTION:
        file_suffix = "full_adaptive"
        config_desc = "完整自适应PINN (数据注入+克里金)"
    elif ENABLE_KRIGING and not ENABLE_DATA_INJECTION:
        file_suffix = "kriging_only"
        config_desc = "仅克里金重采样策略"
    elif not ENABLE_KRIGING and ENABLE_DATA_INJECTION:
        file_suffix = "data_injection_only"
        config_desc = "仅数据注入策略"
    else:
        file_suffix = "periodic_restart"
        config_desc = "仅周期性重启策略"
    
    png_filename = f"mre_comparison_{file_suffix}.png"
    pdf_filename = f"mre_comparison_{file_suffix}.pdf"
    
    plt.savefig(output_dir / png_filename, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / pdf_filename, bbox_inches='tight')
    
    print(f"✅ 对比图已保存到: {output_dir}")
    print(f"   - PNG格式: {output_dir / png_filename}")
    print(f"   - PDF格式: {output_dir / pdf_filename}")
    print(f"   - 实验配置: {config_desc}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main() 