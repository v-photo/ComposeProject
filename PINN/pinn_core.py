"""
Physics-Informed Neural Networks (PINN) for radiation dose simulation
核心PINN训练和分析工具
"""

import os
import numpy as np
import time
import torch
import deepxde as dde
from scipy.interpolate import RegularGridInterpolator

# 常数
EPSILON = 1e-30

class SimulationConfig:
    """Configuration class for simulation parameters"""
    
    def __init__(self, 
                 space_dims=[20.0, 10.0, 10.0],    # [米] 整个模拟世界的物理尺寸 [X, Y, Z]。
                                                   # 定义所考虑空间的总范围。
                                                   # 假设世界坐标系原点为(0,0,0)，由此推导WORLD_MIN/MAX。
                                                   # important: 一定要设置得和输入数据的实际物理尺寸一致，不然会影响模型得预测准确度
                                                   
                 grid_shape=[100, 100, 100],       # [体素] 主数据网格各维度体素数 [Nx, Ny, Nz]。
                                                   # 通常对应输入数据/MC模拟数据的网格形状。
                                                   
                 pinn_grid_shape=None,             # [体素] PINN预测/评估网格各维度体素数 [Nx, Ny, Nz]。
                                                   # 若为None则默认与`grid_shape`相同。可设置为不同值以获得更精细/粗糙的PINN输出。
                                                   
                 source_energy_MeV=30.0,           # [兆电子伏] 源粒子初始能量（如光子或电子）。
                                                   # 主要用于内部蒙特卡洛模拟。
                                                   
                 n_particles=1000000,              # [计数] 模拟粒子总数。
                                                   # 主要用于内部蒙特卡洛模拟以提高统计精度。
                                                   
                 rho_air_kg_m3=1.205,              # [千克/立方米] 主要介质（如空气）的密度。
                                                   # 用于计算体素质量，也可用于推导PINN的k_initial_guess。
                                                   
                 mass_energy_abs_coeff=0.001901,   # [平方米/千克] 介质对给定源能量的质能吸收系数。
                                                   # 用于计算理论衰减系数（PINN的k_initial_guess）。
                                                   
                 energy_cutoff_MeV=0.01):          # [兆电子伏] 内部MC模拟中粒子终止的能量阈值。
                                                   # 当粒子能量低于此值时停止追踪。
        
        # --- 直接赋值或从输入参数处理的变量 ---
        self.SPACE_DIMS_np = np.array(space_dims)              # 物理世界尺寸的numpy数组 [X, Y, Z]。
        self.GRID_SHAPE_np = np.array(grid_shape)              # 主数据网格形状的numpy数组 [Nx, Ny, Nz]。
        self.PINN_GRID_SHAPE_np = np.array(pinn_grid_shape) if pinn_grid_shape is not None else np.array(grid_shape)
                                                               # PINN评估网格形状的numpy数组 [Nx, Ny, Nz]。
        self.SOURCE_ENERGY_MeV = source_energy_MeV             # 源粒子初始能量。
        self.N_PARTICLES = n_particles                         # 模拟粒子总数。
        self.RHO_AIR_kg_m3 = rho_air_kg_m3                     # 介质密度。
        self.MASS_ENERGY_ABS_COEFF_m2_kg = mass_energy_abs_coeff # 质能吸收系数。
        self.ENERGY_CUTOFF_MeV = energy_cutoff_MeV             # MC模拟的粒子能量截止值。
        
        # --- 从上述参数派生的变量 ---
        self.WORLD_MIN_np = -self.SPACE_DIMS_np / 2.0          # [米] 世界边界最小坐标 [Xmin, Ymin, Zmin]，假设以原点为中心。
        self.WORLD_MAX_np = self.SPACE_DIMS_np / 2.0           # [米] 世界边界最大坐标 [Xmax, Ymax, Zmax]，假设以原点为中心。
        self.VOXEL_SIZE_np = self.SPACE_DIMS_np / self.GRID_SHAPE_np 
                                                               # [米] 主数据网格中每个体素的尺寸 [dx, dy, dz]。
        self.VOXEL_VOLUME_m3 = np.prod(self.VOXEL_SIZE_np)     # [立方米] 主数据网格中单个体素的体积。
        self.VOXEL_MASS_kg = self.RHO_AIR_kg_m3 * self.VOXEL_VOLUME_m3 
                                                               # [千克] 主数据网格中单个体素的质量（假设密度均匀）。
        
        # GPU常量初始化标志
        self._gpu_constants_initialized = False
    
    def setup_gpu_constants(self):
        """Setup GPU constants using CuPy (only when needed for internal MC)"""
        if self._gpu_constants_initialized:
            return
            
        try:
            import cupy as cp
            self.SPACE_DIMS_gpu = cp.asarray(self.SPACE_DIMS_np)
            self.GRID_SHAPE_gpu = cp.asarray(self.GRID_SHAPE_np)
            self.WORLD_MIN_gpu = -self.SPACE_DIMS_gpu / 2.0
            self.WORLD_MAX_gpu = self.SPACE_DIMS_gpu / 2.0
            self.VOXEL_SIZE_gpu = self.SPACE_DIMS_gpu / self.GRID_SHAPE_gpu
            self.VOXEL_VOLUME_m3_gpu = cp.prod(self.VOXEL_SIZE_gpu)
            self.VOXEL_MASS_kg_gpu = self.RHO_AIR_kg_m3 * self.VOXEL_VOLUME_m3_gpu
            
            # Convert scalars to GPU
            self.RHO_AIR_kg_m3_gpu = cp.float64(self.RHO_AIR_kg_m3)
            self.MASS_ENERGY_ABS_COEFF_m2_kg_gpu = cp.float64(self.MASS_ENERGY_ABS_COEFF_m2_kg)
            self.ENERGY_CUTOFF_MeV_gpu = cp.float64(self.ENERGY_CUTOFF_MeV)
            self.SOURCE_ENERGY_MeV_gpu = cp.float64(self.SOURCE_ENERGY_MeV)
            
            self._gpu_constants_initialized = True
        except ImportError:
            print("Warning: CuPy not available, GPU constants not initialized")

class PINNTrainer:
    """Physics-Informed Neural Network trainer for dose prediction"""
    
    def __init__(self, physical_params=None):
        """
        Initialize PINN trainer
        
        Args:
            physical_params: Dict with physical parameters like:
                {
                    'rho_material': 1.205,  # kg/m³
                    'mass_energy_abs_coeff': 0.001901,  # m²/kg
                    'k_initial_guess': None  # If None, calculated from above
                }
        """
        if physical_params is None:
            # Default air parameters
            physical_params = {
                'rho_material': 1.205,  # kg/m³ for air
                'mass_energy_abs_coeff': 0.001901,  # m²/kg
            }
        
        self.physical_params = physical_params # important: 物理参数字典，目的是用来存储与模拟相关的物理常数或材料属性。
        self.initial_learned_params_for_validation = {} # important: 新增一个字典来存储初始化参数 
        self.k_initial_guess = self.physical_params.get('k_initial_guess', 
            self.physical_params['rho_material'] * self.physical_params['mass_energy_abs_coeff'])
        
        self.model = None
        self.log_k_pinn = None #important: 辐射衰减系数 （这里修改k_pinn为log_k_pin，目的是为了使迭代过程中k_pinn不为0）
        self.source_params = None
        self.geometry = None
    
    def create_pinn_model(self, dose_data, sampled_points_xyz, sampled_log_doses_values, 
                     include_source=False, network_config=None, source_init_params=None,
                     source_init_method='weighted_centroid', prior_knowledge=None): 
        """
        Create PINN model with flexible data input
        
        Args:
            dose_data: Dict from DataLoader.load_dose_* functions
            sampled_points_xyz: Training point coordinates
            sampled_log_doses_values: Training dose values (log scale)
            include_source: Whether to include source parameterization
            network_config: Dict with network configuration
        """
        print("定义并训练PINN...")
        
        # Geometry from dose data
        world_min = dose_data['world_min']
        world_max = dose_data['world_max']
        self.geometry = dde.geometry.Cuboid(world_min, world_max)
        
        # Network configuration
        if network_config is None:
            if include_source:
                network_config = {'layers': [3] + [128] * 6 + [1], 'activation': 'tanh'}
            else:
                network_config = {'layers': [3] + [64] * 4 + [1], 'activation': 'tanh'}
        
        # Trainable parameters
        safe_k_initial_guess = max(float(self.k_initial_guess), EPSILON)
        self.log_k_pinn = dde.Variable(np.log(safe_k_initial_guess)) # important: 使用对数形式的k_pinn
        
        # 在源项参数创建部分，添加更好的初始化
        if include_source:
            # 源项参数 - 现实的初始化方法
            if source_init_params is None:
                # 将对数剂量转换回线性剂量进行初始化估计
                sampled_doses_linear = np.exp(sampled_log_doses_values)
                source_init_params = self._estimate_source_initial_params(
                    dose_data, sampled_points_xyz, sampled_doses_linear,
                    method=source_init_method, prior_knowledge=prior_knowledge
                )
            
            self.source_params = {
                'xs': dde.Variable(float(source_init_params.get('xs', 0.0))),
                'ys': dde.Variable(float(source_init_params.get('ys', 0.0))),
                'zs': dde.Variable(float(source_init_params.get('zs', 0.0))),
                'As': dde.Variable(float(source_init_params.get('As', 1.0))),
                'log_sigma_s_sq': dde.Variable(float(source_init_params.get('log_sigma_s_sq', np.log(0.5))))
            }
            
            print(f"源项参数初始值 (使用 {source_init_method} 方法):")
            print(f"  位置: ({source_init_params.get('xs', 0.0):.2f}, {source_init_params.get('ys', 0.0):.2f}, {source_init_params.get('zs', 0.0):.2f})")
            print(f"  强度: {source_init_params.get('As', 1.0):.2e}")
            print(f"  扩散: {np.sqrt(np.exp(source_init_params.get('log_sigma_s_sq', np.log(0.5)))):.2f}")
                    
            def pde_with_source(x, u):
                return self._pde_with_source(x, u)
            
            pde_func = pde_with_source
        else:
            def pde_no_source(x, u):
                return self._pde_no_source(x, u)
            
            pde_func = pde_no_source
        
        # important: 在模型可以编译前，记录初始可训练参数
        self.initial_learned_params_for_validation['log_k_pinn_initial'] = self._get_scalar_value(self.log_k_pinn)
        if self.source_params:
            self.initial_learned_params_for_validation['source_params'] = {}
            for name, var in self.source_params.items():
                self.initial_learned_params_for_validation['source_params'][name] = self._get_scalar_value(var)
        
        # Data points
        observe_x = sampled_points_xyz
        observe_y = sampled_log_doses_values.reshape(-1, 1)
        data_points = dde.icbc.PointSetBC(observe_x, observe_y, component=0)
        
        # Create data
        num_domain_points = 20000 if not include_source else 10000
        num_boundary_points = 0 if not include_source else 2000
        
        data = dde.data.PDE(
            self.geometry,
            pde_func,
            [data_points],
            num_domain=num_domain_points,
            num_boundary=num_boundary_points,
            anchors=observe_x
        )
        
        # Neural network
        net = dde.nn.FNN(network_config['layers'], network_config['activation'], "Glorot normal")
        
        self.model = dde.Model(data, net)
        
        return self.model
    
    # important: 定义辅助函数
    def _calculate_derivatives(self, x, u):
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_y = dde.grad.jacobian(u, x, i=0, j=1)
        du_z = dde.grad.jacobian(u, x, i=0, j=2)
        grad_u_sq = du_x**2 + du_y**2 + du_z**2
        
        du_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        du_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)
        laplacian_u = du_xx + du_yy + du_zz
        return grad_u_sq, laplacian_u
    
    def _pde_no_source(self, x, u):
        """PDE without source term"""
        grad_u_sq, laplacian_u = self._calculate_derivatives(x,u)
        # 修改: 从 log_k_pinn 计算 k_squared
        # k_squared = k_pinn**2 = (exp(log_k_pinn))**2 = exp(2 * log_k_pinn)
        k_squared = dde.backend.exp(2 * self.log_k_pinn)
        return grad_u_sq + laplacian_u - k_squared
    
    def _pde_with_source(self, x, u):
        """PDE with source term"""
        grad_u_sq, laplacian_u = self._calculate_derivatives(x,u)
        
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        
        sigma_s_sq = dde.backend.exp(self.source_params['log_sigma_s_sq'])
        sigma_s_sq_safe = dde.backend.relu(sigma_s_sq) + 1e-6 # 避免除以0或过小的值
        
        source_term = self.source_params['As'] * dde.backend.exp(
            -((x_coord - self.source_params['xs'])**2 + 
              (y_coord - self.source_params['ys'])**2 + 
              (z_coord - self.source_params['zs'])**2) / sigma_s_sq_safe
        )
        
        # 修改: 从 log_k_pinn 计算 k_squared
        k_squared = dde.backend.exp(2 * self.log_k_pinn)
        return grad_u_sq + laplacian_u - k_squared - source_term
    
    def train(self, epochs=10000, use_lbfgs=True, loss_weights=None, display_every=500):
        """Train the PINN model with parameter monitoring"""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_pinn_model() first.")
            
        pinn_start_time = time.time()
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = [10, 100] if not self.source_params else [1, 10]
        
        # 收集所有可训练变量
        trainable_variables = [self.log_k_pinn]
        if self.source_params:
            trainable_variables.extend(list(self.source_params.values()))
        
        # Adam training
        print("开始Adam训练...")
        self.model.compile("adam", lr=1e-3, loss_weights=loss_weights, 
                        external_trainable_variables=trainable_variables)
        
        # 打印初始参数
        print("=== 初始参数值 ===")
        self._print_current_parameters()
        
        # 分阶段训练，每个阶段后打印参数
        stages = [epochs // 4, epochs // 2, epochs * 3 // 4, epochs]
        last_epoch = 0
        
        for i, stage in enumerate(stages):
            current_epochs = stage - last_epoch
            if current_epochs > 0:
                print(f"\n训练阶段 {i+1}/4: 第 {last_epoch+1} 到 {stage} epoch...")
                losshistory, train_state = self.model.train(
                    iterations=current_epochs, 
                    display_every=display_every
                )
                
                print(f"\n--- 第 {stage} epoch 参数状态 ---")
                self._print_current_parameters()
                
            last_epoch = stage
        
        # L-BFGS refinement
        if use_lbfgs:
            print("\n=== 切换到L-BFGS优化 ===")
            self.model.compile("L-BFGS", loss_weights=loss_weights, 
                            external_trainable_variables=trainable_variables)
            
            print("开始L-BFGS精细调优...")
            losshistory_lbfgs, train_state_lbfgs = self.model.train(
                display_every=50
            )
            
            print("\n--- L-BFGS优化后参数 ---")
            self._print_current_parameters()
        
        pinn_end_time = time.time()
        print(f"\n=== 训练完成 ===")
        print(f"总耗时: {pinn_end_time - pinn_start_time:.2f} 秒")
        
        self.validate_parameter_updates() # 直接调用，不需要传参了
        
        # 打印最终参数
        print("\n=== 最终学习参数 ===")
        self._print_current_parameters()
    
    def _get_scalar_value(self, variable):
        """Extract scalar value from DeepXDE variable"""
        val = variable.detach().cpu().numpy()
        return val.item() if val.ndim == 0 else val[0]
    
    def predict(self, prediction_points, batch_size=100000):
        """
        Predict dose values at given points, with batching to avoid memory issues.
        
        Args:
            prediction_points: Numpy array of points to predict (N, 3)
            batch_size: Number of points to predict in each batch
        
        Returns:
            Numpy array of predicted dose values (N,)
        """
        if self.model is None:
            raise ValueError("Model is not created. Call create_pinn_model first.")
        
        n_points = prediction_points.shape[0]
        if n_points == 0:
            return np.array([])
            
        print(f"开始预测 {n_points} 个点 (批大小: {batch_size})...")
        
        # 逐批次进行预测
        predictions = []
        for i in range(0, n_points, batch_size):
            batch_points = prediction_points[i:i+batch_size, :]
            
            # 预测的是对数剂量
            log_dose_pred_batch = self.model.predict(batch_points)
            
            # 转换回物理剂量
            dose_pred_batch = np.exp(log_dose_pred_batch)
            predictions.append(dose_pred_batch)
            
            # 打印进度
            print(f"  - 已完成 {min(i + batch_size, n_points)} / {n_points}...", end='\r')

        print("\n✅ 预测完成。")
        
        # 将所有批次结果合并
        predicted_doses = np.vstack(predictions)
        
        return predicted_doses.flatten()
    
    def get_learned_parameters(self):
        """Get the learned parameters"""
        log_k_val = self._get_scalar_value(self.log_k_pinn)
        k_val = np.exp(log_k_val)
        if self.source_params:
            source_vals = {
                'xs': self._get_scalar_value(self.source_params['xs']),
                'ys': self._get_scalar_value(self.source_params['ys']),
                'zs': self._get_scalar_value(self.source_params['zs']),
                'As': self._get_scalar_value(self.source_params['As']),
                'sigma_s': np.sqrt(np.exp(self._get_scalar_value(self.source_params['log_sigma_s_sq'])))
            }
            return k_val, source_vals
        
        return k_val, None

    def _print_current_parameters(self):
        """打印当前参数值的辅助方法"""
        log_k_val = self._get_scalar_value(self.log_k_pinn)
        k_val = np.exp(log_k_val)
        print(f"log_k_pinn: {log_k_val:.6f}  => k_pinn: {k_val:.6f}")
        
        if self.source_params:
            xs_val = self._get_scalar_value(self.source_params['xs'])
            ys_val = self._get_scalar_value(self.source_params['ys'])
            zs_val = self._get_scalar_value(self.source_params['zs'])
            As_val = self._get_scalar_value(self.source_params['As'])
            log_sigma_val = self._get_scalar_value(self.source_params['log_sigma_s_sq'])
            
            print(f"源参数:")
            print(f"  位置: ({xs_val:.3f}, {ys_val:.3f}, {zs_val:.3f})")
            print(f"  强度: {As_val:.2e}")
            print(f"  扩散: {np.sqrt(np.exp(log_sigma_val)):.3f}")

    def _estimate_source_initial_params(self, dose_data, sampled_points_xyz, sampled_doses_values, 
                                       method='weighted_centroid', prior_knowledge=None):
        """
        从有限采样数据中估计源项参数的初始值
        
        Args:
            dose_data: 剂量数据字典
            sampled_points_xyz: 采样点坐标
            sampled_doses_values: 采样点剂量值
            method: 初始化方法 ('weighted_centroid', 'geometric_center', 'prior_based', 'gradient_based')
            prior_knowledge: 先验知识字典，如 {'source_region': [xmin, xmax, ymin, ymax, zmin, zmax]}
        """
        print(f"使用 '{method}' 方法估计源项参数初始值...")
        
        world_min = dose_data['world_min']
        world_max = dose_data['world_max']
        
        if method == 'weighted_centroid':
            # 方法1：基于剂量值的加权质心
            # 使用剂量值作为权重计算质心
            weights = sampled_doses_values + 1e-10  # 避免零权重
            weights = weights / np.sum(weights)  # 归一化
            
            source_pos_init = np.average(sampled_points_xyz, weights=weights, axis=0)
            
        elif method == 'geometric_center':
            # 方法2：采样区域的几何中心
            source_pos_init = (world_min + world_max) / 2.0
            
        elif method == 'prior_based' and prior_knowledge is not None:
            # 方法3：基于先验知识
            if 'source_region' in prior_knowledge:
                region = prior_knowledge['source_region']
                source_pos_init = np.array([
                    (region[0] + region[1]) / 2,  # x中心
                    (region[2] + region[3]) / 2,  # y中心  
                    (region[4] + region[5]) / 2   # z中心
                ])
            else:
                # 降级到几何中心
                source_pos_init = (world_min + world_max) / 2.0
                
        elif method == 'gradient_based':
            # 方法4：基于剂量梯度推测源方向
            try:
                # 计算高剂量区域的质心
                high_dose_threshold = np.percentile(sampled_doses_values, 75)  # 75分位数
                high_dose_mask = sampled_doses_values >= high_dose_threshold
                
                if np.sum(high_dose_mask) >= 3:  # 至少需要3个点
                    high_dose_points = sampled_points_xyz[high_dose_mask]
                    high_dose_values = sampled_doses_values[high_dose_mask]
                    
                    # 加权质心
                    weights = high_dose_values / np.sum(high_dose_values)
                    source_pos_init = np.average(high_dose_points, weights=weights, axis=0)
                else:
                    # 降级到几何中心
                    source_pos_init = (world_min + world_max) / 2.0
                    
            except Exception as e:
                print(f"梯度方法失败，降级到几何中心: {e}")
                source_pos_init = (world_min + world_max) / 2.0
        else:
            # 默认：几何中心
            source_pos_init = (world_min + world_max) / 2.0
        
        # 确保源位置在有效范围内
        source_pos_init = np.clip(source_pos_init, world_min, world_max)
        
        # 估计源强度：基于剂量的统计特性而非最大值
        median_dose = np.median(sampled_doses_values)
        percentile_90_dose = np.percentile(sampled_doses_values, 90)
        
        # 使用90分位数作为参考，这比最大值更稳健
        As_init = percentile_90_dose * 0.5  # 保守估计
        
        # 估计源分布宽度：基于采样点的空间分布
        if len(sampled_points_xyz) > 1:
            # 计算采样点到估计源位置的距离分布
            distances = np.linalg.norm(sampled_points_xyz - source_pos_init, axis=1)
            
            # 使用中位数距离作为特征尺度
            median_distance = np.median(distances)
            sigma_init = max(median_distance * 0.3, 0.1)  # 至少0.1米
            
            # 或者基于高剂量区域的空间扩展
            high_dose_threshold = np.percentile(sampled_doses_values, 60)
            high_dose_mask = sampled_doses_values >= high_dose_threshold
            if np.sum(high_dose_mask) >= 2:
                high_dose_distances = distances[high_dose_mask]
                sigma_init = max(np.std(high_dose_distances), sigma_init)
        else:
            # 单点情况，使用默认值
            domain_size = np.linalg.norm(world_max - world_min)
            sigma_init = domain_size * 0.1  # 域尺寸的10%
        
        # 限制sigma在合理范围内
        domain_characteristic_length = np.min(world_max - world_min)
        sigma_init = np.clip(sigma_init, 0.05, domain_characteristic_length * 0.5)
        
        result = {
            'xs': source_pos_init[0],
            'ys': source_pos_init[1], 
            'zs': source_pos_init[2],
            'As': As_init,
            'log_sigma_s_sq': np.log(sigma_init**2)
        }
        
        print(f"初始化结果:")
        print(f"  源位置: ({result['xs']:.3f}, {result['ys']:.3f}, {result['zs']:.3f}) m")
        print(f"  源强度: {result['As']:.3e}")
        print(f"  源扩散: {sigma_init:.3f} m") 
        print(f"  采样统计: 中位数剂量={median_dose:.3e}, 90分位数剂量={percentile_90_dose:.3e}")
        
        return result

    def validate_parameter_updates(self): # 不再需要 initial_params 参数
        if not self.initial_learned_params_for_validation:
            print("错误: 初始可训练参数未被记录。请确保在 create_pinn_model 中记录。")
            return
        
        initial_params = self.initial_learned_params_for_validation # 使用内部存储的初始值
        # ... 后续验证逻辑使用 initial_params ...
        print("\n=== 参数更新验证 ===")
        current_log_k = self._get_scalar_value(self.log_k_pinn)
        initial_log_k = initial_params['log_k_pinn_initial']
        
        print(f"log_k_pinn: {initial_log_k:.6f} -> {current_log_k:.6f} (变化: {abs(current_log_k - initial_log_k):.6f})")
        print(f"  k_pinn (derived): {np.exp(initial_log_k):.6f} -> {np.exp(current_log_k):.6f}")
        
        if self.source_params and 'source_params' in initial_params:
            print("源项参数:")
            initial_source_params_dict = initial_params['source_params']
            for name, var in self.source_params.items():
                current_val = self._get_scalar_value(var)
                if name in initial_source_params_dict:
                    initial_val = initial_source_params_dict[name]
                    change = abs(current_val - initial_val)
                    print(f"  {name}: {initial_val:.6f} -> {current_val:.6f} (变化: {change:.6f})")
                else: # 这种情况理论上不应该发生，如果初始值都记录了的话
                    print(f"  {name}: (无初始值记录 for {name}) -> {current_val:.6f}")
        return True

class ResultAnalyzer:
    """Analyzer for comparing PINN and Monte Carlo results"""
    
    @staticmethod
    def evaluate_predictions(dose_pinn_grid, dose_mc_data, pinn_grid_coords, 
                           sampled_points_xyz=None, sampled_doses_values=None):
        """
        Evaluate PINN predictions against Monte Carlo truth
        
        Args:
            dose_pinn_grid: PINN prediction grid
            dose_mc_data: MC dose data dict from DataLoader
            pinn_grid_coords: Coordinates for PINN prediction grid
            sampled_points_xyz: Training point coordinates (optional)
            sampled_doses_values: Training dose values (optional)
        
        Returns:
            dict: Evaluation results
        """
        print("\nPINN预测与评估...")
        
        dose_grid_mc = dose_mc_data['dose_grid']
        
        # Interpolate PINN to MC grid for fair comparison
        pinn_interp_func = RegularGridInterpolator(
            pinn_grid_coords,
            dose_pinn_grid,
            bounds_error=False,
            fill_value=EPSILON
        )
        
        # MC grid coordinates
        mc_grid_shape = dose_mc_data['grid_shape']
        world_min = dose_mc_data['world_min']
        voxel_size = dose_mc_data['voxel_size']
        
        mc_x_centers = world_min[0] + (np.arange(mc_grid_shape[0]) + 0.5) * voxel_size[0]
        mc_y_centers = world_min[1] + (np.arange(mc_grid_shape[1]) + 0.5) * voxel_size[1]
        mc_z_centers = world_min[2] + (np.arange(mc_grid_shape[2]) + 0.5) * voxel_size[2]
        
        MC_XX, MC_YY, MC_ZZ = np.meshgrid(mc_x_centers, mc_y_centers, mc_z_centers, indexing='ij')
        mc_eval_points = np.vstack((MC_XX.ravel(), MC_YY.ravel(), MC_ZZ.ravel())).T
        
        dose_pinn_on_mc_grid_flat = pinn_interp_func(mc_eval_points)
        dose_pinn_on_mc_grid = dose_pinn_on_mc_grid_flat.reshape(mc_grid_shape)
        
        # Calculate errors
        # important(在这里可以设定只考虑剂量率大于前百分之几最大值的数据，默认设置为1%)
        # valid_comparison_mask = (dose_grid_mc > (0.01 * np.max(dose_grid_mc))) #这个用于测试对所有高剂量率点处的预测效果
        valid_comparison_mask = (dose_grid_mc < np.max(dose_grid_mc))
        diff_abs = np.abs(dose_pinn_on_mc_grid[valid_comparison_mask] - dose_grid_mc[valid_comparison_mask])
        relative_error = diff_abs / (np.abs(dose_grid_mc[valid_comparison_mask]) + EPSILON)
        mean_relative_error = np.mean(relative_error)
        
        mae = np.mean(np.abs(dose_pinn_on_mc_grid - dose_grid_mc))
        rmse = np.sqrt(np.mean((dose_pinn_on_mc_grid - dose_grid_mc)**2))
        
        print(f"PINN Mean Relative Error on MC grid: {mean_relative_error:.4%}")
        print(f"PINN MAE on MC grid: {mae:.4e}")
        print(f"PINN RMSE on MC grid: {rmse:.4e}")
        
        # Training point evaluation if provided
        training_results = {}
        if sampled_points_xyz is not None and sampled_doses_values is not None:
            # Evaluate on training points
            dose_pinn_on_samples = pinn_interp_func(sampled_points_xyz)
            mre_training = np.mean(np.abs(dose_pinn_on_samples - sampled_doses_values) / 
                                 (np.abs(sampled_doses_values) + EPSILON))
            training_results = {
                'training_mre': mre_training,
                'dose_pinn_on_samples': dose_pinn_on_samples
            }
            print(f"PINN Mean Relative Error on training points: {mre_training:.4%}")
        
        return {
            'mean_relative_error': mean_relative_error,
            'mae': mae,
            'rmse': rmse,
            'dose_pinn_on_mc_grid': dose_pinn_on_mc_grid,
            'mc_grid_coords': (mc_x_centers, mc_y_centers, mc_z_centers),
            **training_results
        }

def setup_deepxde_backend():
    """Setup DeepXDE backend"""
    os.environ['DDE_BACKEND'] = 'pytorch' 