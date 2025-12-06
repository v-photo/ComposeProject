"""
数据加载与处理模块
Module for data loading and processing.
"""
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import os

# 我们的新环境模块会处理路径和依赖检查
# This centralized approach is cleaner.
from ..utils.environment import PINN_AVAILABLE

if not PINN_AVAILABLE:
    # 如果环境检查失败，立即抛出异常
    raise ImportError("PINN 模块无法加载。请检查项目结构和依赖。")

# 既然PINN_AVAILABLE为True，说明路径已设置，可以直接导入
from data_processing import DataLoader
from dataAnalysis import get_data


class AdaptiveDataLoader:
    """
    一个数据加载器，用于从外部文件加载初始训练数据，并支持灵活的数据集分割策略。
    (原名 DummyDataLoader)
    """
    def __init__(self, data_path: str, space_dims: np.ndarray, num_samples: int):
        self.data_path = data_path
        self.space_dims = space_dims
        self.num_samples = num_samples
        print(f"INFO: (DataLoader) Initialized with data_path='{self.data_path}'")

    def get_training_data(self, split_ratios: Optional[List[float]] = None, test_set_size: Optional[int] = None) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, Dict]:
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
        
        Returns:
            - main_train_set (np.ndarray)
            - reserve_pools (List[np.ndarray])
            - test_set (np.ndarray)
            - dose_data (Dict)
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
                if test_size_fraction < 1e-10:
                    test_size_fraction = 0.0
                elif test_size_fraction > 1.0:
                    test_size_fraction = 1.0
                
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
                if len(remaining_data) == 0 and len(all_sampled_data) > 0:
                     # 如果经过分割后没有剩下任何数据点作为测试集，这是一个潜在问题
                    print("WARNING: No data left for the test set after splitting according to split_ratios.")
                    # 根据场景，可以创建一个空的测试集或抛出错误
                    test_set = np.array([]).reshape(0, all_sampled_data.shape[1])
                else:
                    test_set = remaining_data
        
        print(f"INFO: (DataLoader) ✅ Split data into: Main training ({len(main_train_set)}), Test ({len(test_set)}), Reserve Pools ({len(reserve_pools)} pools).")
        if reserve_pools:
            for i, pool in enumerate(reserve_pools):
                print(f"    - Reserve Pool {i+1}: {len(pool)} points")

        return main_train_set, reserve_pools, test_set, dose_data

    def _generate_independent_test_set(self, dose_data: dict, test_set_size: int) -> np.ndarray:
        """
        生成完全独立于训练数据的测试集，在整个物理域内均匀采样。
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

def load_data_from_xlsx(
    file_path: str,
    column_map: Dict[str, str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    从 .xlsx 文件加载数据，并将其划分为训练集和测试集。

    Args:
        file_path (str): .xlsx文件的路径。
        column_map (Dict[str, str]): 将标准名称映射到文件中实际列名的字典。
                                     需要包含 'x', 'y', 'z', 'value'。
        test_size (float): 用于测试集的数据比例。
        random_state (int): 随机种子，用于可复现的划分。

    Returns:
        A tuple containing:
        - train_points (np.ndarray): 训练点的坐标 (N, 3)。
        - train_values (np.ndarray): 训练点的值 (N, 1)。
        - test_data (np.ndarray): 测试数据，包含坐标和值 (M, 4)。
        - dose_data (Dict): 包含数据边界信息的字典。
    """
    print(f"\n--- 💾 正在从 {file_path} 加载数据 ---")
    
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 数据文件未找到于 '{file_path}'。请确认文件路径是否正确。")
    except Exception as e:
        raise IOError(f"错误: 读取 {file_path} 时出错: {e}")

    # 根据映射重命名列
    try:
        df = df.rename(columns={v: k for k, v in column_map.items()})
        required_cols = {'x', 'y', 'z', 'value'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"错误: 文件中缺少必要的列: {missing} (已根据列映射重命名)")
    except Exception as e:
         raise ValueError(f"错误: 应用列映射时出错: {e}")

    all_points = df[['x', 'y', 'z']].values
    all_values = df[['value']].values

    # 划分数据
    train_points, test_points, train_values, test_values = train_test_split(
        all_points, all_values, test_size=test_size, random_state=random_state
    )
    
    test_data = np.hstack([test_points, test_values])

    # 计算数据边界
    dose_data = {
        'world_min': all_points.min(axis=0),
        'world_max': all_points.max(axis=0),
        'space_dims': all_points.max(axis=0) - all_points.min(axis=0)
    }
    
    print(f"  ✅ 数据加载完毕: {len(train_points)}个训练点, {len(test_points)}个测试点。")
    
    return train_points, train_values.flatten(), test_data, dose_data

def load_and_process_data(file_path: str, column_map: Dict[str, str]) -> Dict[str, Any]:
    """
    从 .xlsx 文件加载原始数据，并将其处理成包含剂量网格和空间信息的标准化字典。

    Args:
        file_path (str): .xlsx文件的路径。
        column_map (Dict[str, str]): 标准名称到文件中实际列名的映射。

    Returns:
        一个包含完整数据信息的字典 (dose_data)。
    """
    print(f"\n--- 💾 正在从 {file_path} 加载并处理数据 ---")
    
    try:
        df = pd.read_excel(file_path, header=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 数据文件未找到于 '{file_path}'。")
    
    df = df.rename(columns={v: k for k, v in column_map.items()})
    required_cols = {'x', 'y', 'z', 'value'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"错误: 文件中缺少必要的列: {missing}")

    # 从数据中推断网格结构
    x_coords = np.unique(df['x'])
    y_coords = np.unique(df['y'])
    z_coords = np.unique(df['z'])
    
    grid_shape = (len(x_coords), len(y_coords), len(z_coords))
    
    # 确保数据点数量与网格大小匹配
    if len(df) != np.prod(grid_shape):
        raise ValueError(f"数据点总数 ({len(df)})与推断的网格大小 {grid_shape} 不匹配。")
        
    # 将DataFrame重塑为3D剂量网格
    df_sorted = df.sort_values(by=['z', 'y', 'x'])
    dose_grid = df_sorted['value'].values.reshape(grid_shape, order='F')

    world_min = df[['x', 'y', 'z']].min().values
    world_max = df[['x', 'y', 'z']].max().values
    
    dose_data = {
        'dose_grid': dose_grid,
        'world_min': world_min,
        'world_max': world_max,
        'space_dims': world_max - world_min,
        'voxel_size': (world_max - world_min) / (np.array(grid_shape) - 1),
        'grid_shape': grid_shape
    }
    
    print(f"  ✅ 数据处理完毕。网格尺寸: {grid_shape}。")
    return dose_data

def load_3d_data_from_sheets(
    file_path: str,
    sheet_name_template: str,
    use_cols: str,
    z_size: int,
    y_size: int
) -> np.ndarray:
    """
    从一个Excel文件的多个sheets中加载数据，并将其组装成一个3D Numpy数组。
    此函数复现了 'PINN/dataAnalysis.py' 中 get_data 的核心逻辑，包括Pickle缓存。

    返回:
        一个三维Numpy数组，代表剂量网格 (dose_grid)。
    """
    print(f"\n--- 💾 正在从 {file_path} 的多个Sheets加载3D数据 ---")
    
    p = Path(file_path)
    cache_dir = p.parent / f"{p.stem}_pkl_data"
    cache_dir.mkdir(exist_ok=True)
    
    data_sheets = {}
    use_cache = all((cache_dir / f"pkl{z}.pkl").exists() for z in range(z_size))

    if use_cache:
        print(f"  - 正在从缓存目录加载: {cache_dir}")
        for z in range(z_size):
            df = pd.read_pickle(cache_dir / f"pkl{z}.pkl")
            data_sheets[z] = df.values
    else:
        print(f"  - 正在从Excel文件读取并创建缓存...")
        for z in range(z_size):
            sheet_name = sheet_name_template.replace("z", str(z + 1))
            try:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=None,
                    usecols=use_cols,
                    names=list(range(y_size))
                )
                pd.to_pickle(df, cache_dir / f"pkl{z}.pkl")
                data_sheets[z] = df.values
                if z % 10 == 0: print(f"    ...已处理 {z+1}/{z_size} 个sheets")
            except Exception as e:
                raise IOError(f"读取 Sheet '{sheet_name}' 时出错: {e}")

    # 将所有2D切片堆叠成一个3D数组
    # 原始数据[z][y][x]，我们需要[x][y][z]
    dose_grid_zyx = np.stack(list(data_sheets.values()), axis=0)
    dose_grid = np.transpose(dose_grid_zyx, (2, 1, 0))
    
    print(f"  ✅ 3D数据加载完毕。最终网格尺寸: {dose_grid.shape}")
    return dose_grid

def process_grid_to_dose_data(
    dose_grid: np.ndarray, 
    space_dims: Tuple[float, float, float] = (20.0, 10.0, 10.0) # 假设值，应从配置获取
) -> Dict[str, Any]:
    """
    将加载的剂量网格处理成包含物理空间信息的标准化字典。
    """
    grid_shape = np.array(dose_grid.shape)
    
    # 假设原点在[0,0,0]
    world_min = np.array([0., 0., 0.])
    world_max = np.array(space_dims)

    dose_data = {
        'dose_grid': dose_grid,
        'world_min': world_min,
        'world_max': world_max,
        'space_dims': world_max - world_min,
        'voxel_size': (world_max - world_min) / (grid_shape - 1),
        'grid_shape': grid_shape
    }
    return dose_data

def sample_training_points(
    dose_data: Dict, num_samples: int, strategy: str = 'positive_only'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从处理过的数据中采样训练点。
    """
    if strategy != 'positive_only':
        raise NotImplementedError("目前仅支持 'positive_only' 采样策略。")

    positive_indices = np.argwhere(dose_data['dose_grid'] > 1e-10) # 仅在有剂量处采样
    
    if len(positive_indices) < num_samples:
        print(f"警告: 请求采样 {num_samples} 点, 但只有 {len(positive_indices)} 个正剂量点可用。")
        num_samples = len(positive_indices)

    sample_indices = positive_indices[
        np.random.choice(len(positive_indices), num_samples, replace=False)
    ]
    
    # 将网格索引转换回世界坐标
    train_points = dose_data['world_min'] + sample_indices * dose_data['voxel_size']
    train_values = dose_data['dose_grid'][
        sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]
    ]

    return train_points, train_values.reshape(-1, 1)

def create_prediction_grid(dose_data: Dict, downsample_factor: int = 1) -> np.ndarray:
    """
    根据降采样系数创建用于全场预测的坐标网格。
    """
    grid_shape = dose_data['grid_shape']
    
    if downsample_factor > 1:
        print(f"⚠️  预测网格将以系数 {downsample_factor} 进行降采样。")
        step = int(downsample_factor)
        x_indices = np.arange(0, grid_shape[0], step)
        y_indices = np.arange(0, grid_shape[1], step)
        z_indices = np.arange(0, grid_shape[2], step)
    else:
        x_indices = np.arange(grid_shape[0])
        y_indices = np.arange(grid_shape[1])
        z_indices = np.arange(grid_shape[2])

    pred_x = dose_data['world_min'][0] + x_indices * dose_data['voxel_size'][0]
    pred_y = dose_data['world_min'][1] + y_indices * dose_data['voxel_size'][1]
    pred_z = dose_data['world_min'][2] + z_indices * dose_data['voxel_size'][2]
    
    XX, YY, ZZ = np.meshgrid(pred_x, pred_y, pred_z, indexing='ij')
    prediction_points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
    
    return prediction_points


# ==================== Kriging 风格采样扩展 ====================

def sample_kriging_style(
    dose_data: Dict,
    box_origin: List[int] = [5, 5, 5],
    box_extent: List[int] = [90, 90, 90],
    step_sizes: List[int] = [5],
    source_positions: List[List[int]] = None,
    source_exclusion_radius: float = 30.0,
    return_dataframe: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Kriging风格的结构化网格采样（与 Kriging/dataAnalysis.py 的 training_sampling 一致）
    
    此函数是对 unified_sampling.py 的简化封装，提供与 sample_training_points 一致的接口。
    
    Args:
        dose_data: PINN格式的数据字典
        box_origin: 采样区域起点 [x, y, z] (网格索引)
        box_extent: 采样区域在各方向的延伸长度 [x_len, y_len, z_len]
        step_sizes: 采样步长列表
        source_positions: 源点位置列表，用于排除源点附近区域
        source_exclusion_radius: 源点排除半径
        return_dataframe: 是否同时返回DataFrame
        
    Returns:
        (train_points, train_values) 或 (train_points, train_values, df) 如果 return_dataframe=True
    """
    dose_grid = dose_data['dose_grid']
    world_min = dose_data.get('world_min', np.zeros(3))
    voxel_size = dose_data.get('voxel_size', np.ones(3))
    grid_shape = np.array(dose_grid.shape)
    
    if source_positions is None:
        source_positions = []
    
    sampled_data = []
    
    for step in step_sizes:
        x_range = box_extent[0] // step
        y_range = box_extent[1] // step
        z_range = box_extent[2] // step
        
        for xi in range(0, x_range + 1):
            for yi in range(0, y_range + 1):
                for zi in range(0, z_range + 1):
                    x_coord = box_origin[0] + xi * step
                    y_coord = box_origin[1] + yi * step
                    z_coord = box_origin[2] + zi * step
                    
                    # 边界检查
                    if (x_coord >= grid_shape[0] or y_coord >= grid_shape[1] or 
                        z_coord >= grid_shape[2] or x_coord < 0 or y_coord < 0 or z_coord < 0):
                        continue
                    
                    # 检查是否需要排除源点附近
                    skip = False
                    for pos in source_positions:
                        distance = np.sqrt((x_coord - pos[0])**2 + 
                                         (y_coord - pos[1])**2 + 
                                         (z_coord - pos[2])**2)
                        if distance <= source_exclusion_radius:
                            skip = True
                            break
                    
                    if skip:
                        continue
                    
                    value = dose_grid[x_coord, y_coord, z_coord]
                    if value > 1e-10:  # 只采样正剂量点
                        sampled_data.append((x_coord, y_coord, z_coord, value))
    
    if len(sampled_data) == 0:
        raise ValueError("Kriging风格采样未能获取到任何有效点，请检查参数设置")
    
    # 去重
    sampled_data = list(set(sampled_data))
    sampled_data = np.array(sampled_data)
    
    sampled_indices = sampled_data[:, :3].astype(int)
    sampled_values = sampled_data[:, 3]
    
    # 转换为物理坐标
    train_points = world_min + sampled_indices * voxel_size + voxel_size / 2.0
    train_values = sampled_values.reshape(-1, 1)
    
    print(f"Kriging风格采样完成: {len(train_points)} 个训练点")
    
    if return_dataframe:
        df = pd.DataFrame({
            'x': sampled_indices[:, 0],
            'y': sampled_indices[:, 1],
            'z': sampled_indices[:, 2],
            'target': sampled_values
        })
        return train_points, train_values, df
    
    return train_points, train_values
