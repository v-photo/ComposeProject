"""
统一采样模块 (Unified Sampling Module)

此模块提供与 Kriging/dataAnalysis.py 一致的采样方式，
用于统一 PINN 和 Kriging 算法的数据采样流程。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SamplingConfig:
    """采样配置类"""
    center_x: int = 50
    center_y: int = 50
    center_z: int = 50
    inner_radius_max: int = 30
    use_vertices: bool = False
    inner_radius_list: List[int] = field(default_factory=lambda: list(range(1, 3)))
    use_box_area: bool = True
    box_origin: List[int] = field(default_factory=lambda: [5, 5, 5])
    box_extent: List[int] = field(default_factory=lambda: [90, 90, 60])
    step_sizes: List[int] = field(default_factory=lambda: [5])
    x_y_reverse: bool = True
    direction: str = "3vector"
    source_positions: List[List[int]] = field(default_factory=list)
    source_exclusion_radius: float = 10.0


class UnifiedSampler:
    """统一采样器类"""
    
    def __init__(self, data: Dict[int, pd.DataFrame] = None, dose_data: Dict = None):
        self.kriging_data = data
        self.dose_data = dose_data
        
        if dose_data is not None:
            self.grid_shape = np.array(dose_data.get('grid_shape', 
                                           dose_data['dose_grid'].shape))
            self.world_min = dose_data.get('world_min', np.zeros(3))
            self.voxel_size = dose_data.get('voxel_size', np.ones(3))
    
    def training_sampling(self, config: SamplingConfig) -> pd.DataFrame:
        """结构化网格训练采样"""
        if self.kriging_data is None and self.dose_data is None:
            raise ValueError("需要提供 kriging_data 或 dose_data")
        
        sampled_data = []
        
        # 获取数据边界
        if self.kriging_data is not None:
            max_z = len(self.kriging_data)
            first_layer = list(self.kriging_data.values())[0]
            if hasattr(first_layer, 'shape'):
                max_y, max_x = first_layer.shape  # DataFrame shape is (rows, cols) = (y, x)
            else:
                max_y = len(first_layer)
                max_x = len(first_layer[0]) if max_y > 0 else 0
        elif self.dose_data is not None:
            max_x, max_y, max_z = self.grid_shape
        
        # 网格步进采样
        for step in config.step_sizes:
            if config.use_box_area:
                x_range = config.box_extent[0] // step
                y_range = config.box_extent[1] // step
                z_range = config.box_extent[2] // step
                
                for xi in range(0, x_range + 1):
                    for yi in range(0, y_range + 1):
                        for zi in range(0, z_range + 1):
                            x_coord = config.box_origin[0] + xi * step
                            y_coord = config.box_origin[1] + yi * step
                            z_coord = config.box_origin[2] + zi * step
                            
                            # 边界检查
                            if x_coord >= max_x or y_coord >= max_y or z_coord >= max_z:
                                continue
                            if x_coord < 0 or y_coord < 0 or z_coord < 0:
                                continue
                            
                            # 源点排除
                            if self._should_skip_source(x_coord, y_coord, z_coord,
                                                       config.source_positions,
                                                       config.source_exclusion_radius):
                                continue
                            
                            value = self._get_value(x_coord, y_coord, z_coord, config.x_y_reverse)
                            if value is not None:
                                sampled_data.append((x_coord, y_coord, z_coord, value))
            else:
                x_range = config.inner_radius_max // step
                y_range = config.inner_radius_max // step
                z_range = config.inner_radius_max // step
                
                if config.direction == "6vector":
                    x_start, y_start, z_start = -x_range + 1, -y_range + 1, -z_range + 1
                else:
                    x_start, y_start, z_start = 0, 0, 0
                
                for xi in range(x_start, x_range + 1):
                    for yi in range(y_start, y_range + 1):
                        for zi in range(z_start, z_range + 1):
                            x_coord = config.center_x + xi * step
                            y_coord = config.center_y + yi * step
                            z_coord = config.center_z + zi * step
                            
                            if x_coord >= max_x or y_coord >= max_y or z_coord >= max_z:
                                continue
                            if x_coord < 0 or y_coord < 0 or z_coord < 0:
                                continue
                            
                            if self._should_skip_source(x_coord, y_coord, z_coord,
                                                       config.source_positions,
                                                       config.source_exclusion_radius):
                                continue
                            
                            value = self._get_value(x_coord, y_coord, z_coord, config.x_y_reverse)
                            if value is not None:
                                sampled_data.append((x_coord, y_coord, z_coord, value))
        
        df = pd.DataFrame(sampled_data, columns=['x', 'y', 'z', 'target'])
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        print(f"INFO: (UnifiedSampler) 采样完成，共 {len(df)} 个训练点")
        return df
    
    def _get_value(self, x: int, y: int, z: int, x_y_reverse: bool) -> Optional[float]:
        """
        从数据中获取指定坐标的值
        
        重要：使用与 Kriging/dataAnalysis.py 完全一致的访问方式
        原始代码: data[z_coord][y_coord][x_coord] 当 x_y_reverse=True
        即: DataFrame[column_name][row_index]
        """
        try:
            if self.kriging_data is not None:
                if z not in self.kriging_data:
                    return None
                layer = self.kriging_data[z]
                
                # 使用与原始 training_sampling 完全一致的访问方式
                # data[z][y][x] 意味着: DataFrame[column_y][row_x]
                if x_y_reverse:
                    # 原始: data[z_coord][y_coord][x_coord]
                    # DataFrame 列名是数字，所以 layer[y] 选择列 y，然后 [x] 选择行 x
                    return layer[y][x]
                else:
                    # 原始: data[z_coord][x_coord][y_coord]
                    return layer[x][y]
                    
            elif self.dose_data is not None:
                dose_grid = self.dose_data['dose_grid']
                if (0 <= x < dose_grid.shape[0] and 
                    0 <= y < dose_grid.shape[1] and 
                    0 <= z < dose_grid.shape[2]):
                    return dose_grid[x, y, z]
                return None
        except (KeyError, IndexError):
            return None
        return None
    
    def _should_skip_source(self, x: int, y: int, z: int,
                           source_positions: List[List[int]],
                           exclusion_radius: float) -> bool:
        if not source_positions:
            return False
        for pos in source_positions:
            distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)
            if distance <= exclusion_radius:
                return True
        return False
    
    def convert_to_physical_coords(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """将网格索引转换为物理坐标"""
        if self.dose_data is None:
            raise ValueError("需要 dose_data 来进行坐标转换")
        
        indices = df[['x', 'y', 'z']].values
        values = df['target'].values
        points_xyz = self.world_min + indices * self.voxel_size + self.voxel_size / 2.0
        return points_xyz, values


def create_default_sampling_config(
    box_origin: List[int] = [5, 5, 5],
    box_extent: List[int] = [90, 90, 60],
    step_sizes: List[int] = [5],
    source_positions: List[List[int]] = None,
    source_exclusion_radius: float = 30.0
) -> SamplingConfig:
    return SamplingConfig(
        use_box_area=True,
        box_origin=box_origin,
        box_extent=box_extent,
        step_sizes=step_sizes,
        x_y_reverse=True,
        direction="3vector",
        source_positions=source_positions or [],
        source_exclusion_radius=source_exclusion_radius
    )


def sample_like_kriging(
    dose_data: Dict,
    config: SamplingConfig = None,
    return_physical_coords: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if config is None:
        grid_shape = np.array(dose_data['dose_grid'].shape)
        config = create_default_sampling_config(
            box_origin=[5, 5, 5],
            box_extent=[int(grid_shape[0] - 10), int(grid_shape[1] - 10), int(grid_shape[2] - 10)],
            step_sizes=[5]
        )
    
    sampler = UnifiedSampler(dose_data=dose_data)
    df = sampler.training_sampling(config)
    
    if return_physical_coords:
        train_points, train_values = sampler.convert_to_physical_coords(df)
    else:
        train_points = df[['x', 'y', 'z']].values
        train_values = df['target'].values
    
    return train_points, train_values, df
