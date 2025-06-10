"""
辐射数据处理和加载工具
支持多种输入格式的辐射场数据处理
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, List

# 常数
EPSILON = 1e-30

class RadiationDataProcessor:
    """
    Enhanced data processor for radiation field data
    Supports multiple input formats including {z: DataFrame[y, x]} from tool.py
    """
    
    def __init__(self, space_dims=None, world_bounds=None):
        """
        Initialize the data processor
        
        Args:
            space_dims: Physical dimensions [x, y, z] in meters
            world_bounds: Dict with 'min' and 'max' arrays, or None for auto-detection
        """
        self.space_dims = np.array(space_dims) if space_dims is not None else None
        self.world_bounds = world_bounds
        self.dose_data = None
        
    def load_from_dict(self, data_dict: Dict, space_dims=None, world_bounds=None):
        """
        Load radiation data from dictionary format {z: DataFrame[y, x]} or {z: numpy_array}
        Compatible with tool.py RadiationDataset format
        
        Args:
            data_dict: Dictionary where keys are z-coordinates and values are 2D data (DataFrame or numpy array)
            space_dims: Physical dimensions [x, y, z] in meters
            world_bounds: Physical bounds dict with 'min' and 'max' keys
            
        Returns:
            dict: Standardized dose_data format for PINN usage
        """
        print("Loading radiation data from dictionary format...")
        
        # Extract and sort z coordinates
        z_coords = sorted(data_dict.keys())
        print(f"Found {len(z_coords)} z-layers: {z_coords}")
        
        # Get first layer to determine dimensions
        first_layer = data_dict[z_coords[0]]
        
        # Convert DataFrame or array to numpy
        if hasattr(first_layer, 'values'):  # DataFrame
            first_array = first_layer.values
            print("Detected pandas DataFrame format")
        elif hasattr(first_layer, 'shape'):  # numpy array
            first_array = np.array(first_layer)
            print("Detected numpy array format")
        else:
            first_array = np.array(first_layer)
            print("Converting to numpy array format")
        
        y_size, x_size = first_array.shape
        z_size = len(z_coords)
        
        print(f"Data dimensions: X={x_size}, Y={y_size}, Z={z_size}")
        
        # Create 3D dose grid
        dose_grid = np.zeros((x_size, y_size, z_size), dtype=np.float64)
        
        for z_idx, z_coord in enumerate(z_coords):
            layer_data = data_dict[z_coord]
            
            # Convert to numpy if needed
            if hasattr(layer_data, 'values'):
                layer_array = layer_data.values
            else:
                layer_array = np.array(layer_data)
            
            # Note: transpose to match expected grid convention (x, y, z)
            dose_grid[:, :, z_idx] = layer_array.T
        
        # Handle physical dimensions and bounds
        if space_dims is not None:
            self.space_dims = np.array(space_dims)
        elif self.space_dims is None:
            # Default physical dimensions if not provided
            self.space_dims = np.array([20.0, 10.0, 10.0])  # meters
            print(f"Using default space dimensions: {self.space_dims}")
        
        if world_bounds is not None:
            self.world_bounds = world_bounds
            world_min = np.array(world_bounds['min'])
            world_max = np.array(world_bounds['max'])
        elif self.world_bounds is not None:
            world_min = np.array(self.world_bounds['min'])
            world_max = np.array(self.world_bounds['max'])
        else:
            # Default: centered around origin
            world_min = -self.space_dims / 2.0
            world_max = self.space_dims / 2.0
            print(f"Using default world bounds: {world_min} to {world_max}")
        
        # Calculate derived parameters
        grid_shape = np.array([x_size, y_size, z_size])
        voxel_size = (world_max - world_min) / grid_shape
        
        # Create standardized dose_data format
        self.dose_data = {
            'dose_grid': dose_grid,
            'world_min': world_min,
            'world_max': world_max,
            'voxel_size': voxel_size,
            'grid_shape': grid_shape,
            'space_dims': self.space_dims,
            'z_coords': np.array(z_coords),
            'original_data_dict': data_dict  # Keep reference to original data
        }
        
        # Statistics
        nonzero_count = np.count_nonzero(dose_grid)
        total_count = dose_grid.size
        print(f"Data statistics:")
        print(f"  - Total voxels: {total_count:,}")
        print(f"  - Non-zero voxels: {nonzero_count:,} ({nonzero_count/total_count*100:.2f}%)")
        print(f"  - Value range: {np.min(dose_grid):.2e} to {np.max(dose_grid):.2e}")
        print(f"  - Voxel size: {voxel_size}")
        
        return self.dose_data
    
    def load_from_numpy(self, dose_array, space_dims, world_bounds=None):
        """
        Load radiation data from 3D numpy array
        
        Args:
            dose_array: 3D numpy array (x, y, z)
            space_dims: Physical dimensions [x, y, z] in meters
            world_bounds: Physical bounds dict or None for centered
            
        Returns:
            dict: Standardized dose_data format
        """
        print("Loading radiation data from numpy array...")
        
        if dose_array.ndim != 3:
            raise ValueError(f"Expected 3D array, got {dose_array.ndim}D")
        
        self.space_dims = np.array(space_dims)
        grid_shape = np.array(dose_array.shape)
        
        if world_bounds is not None:
            world_min = np.array(world_bounds['min'])
            world_max = np.array(world_bounds['max'])
        else:
            world_min = -self.space_dims / 2.0
            world_max = self.space_dims / 2.0
        
        voxel_size = (world_max - world_min) / grid_shape
        
        self.dose_data = {
            'dose_grid': dose_array.astype(np.float64),
            'world_min': world_min,
            'world_max': world_max,
            'voxel_size': voxel_size,
            'grid_shape': grid_shape,
            'space_dims': self.space_dims
        }
        
        print(f"Loaded numpy array: shape {dose_array.shape}, range {np.min(dose_array):.2e} to {np.max(dose_array):.2e}")
        
        return self.dose_data
    
    def get_dose_data(self):
        """Get the standardized dose_data format"""
        if self.dose_data is None:
            raise ValueError("No data loaded. Call load_from_dict() or load_from_numpy() first.")
        return self.dose_data
    
    def convert_to_tool_format(self):
        """
        Convert current dose_data back to tool.py compatible format {z: DataFrame}
        
        Returns:
            dict: Dictionary in {z: DataFrame[y, x]} format
        """
        if self.dose_data is None:
            raise ValueError("No data loaded.")
        
        dose_grid = self.dose_data['dose_grid']
        z_coords = self.dose_data.get('z_coords', range(dose_grid.shape[2]))
        
        result_dict = {}
        for z_idx, z_coord in enumerate(z_coords):
            # Extract z-slice and transpose back to (y, x) format
            z_slice = dose_grid[:, :, z_idx].T
            result_dict[z_coord] = pd.DataFrame(z_slice)
        
        return result_dict
    
    def normalize_data(self, method='robust'):
        """
        Normalize the dose data using different methods
        
        Args:
            method: 'robust' (quantile-based), 'minmax', or 'standard'
        """
        if self.dose_data is None:
            raise ValueError("No data loaded.")
        
        dose_grid = self.dose_data['dose_grid']
        
        if method == 'robust':
            # Use quantile-based normalization like in tool.py
            q01 = np.quantile(dose_grid, 0.01)
            q99 = np.quantile(dose_grid, 0.99)
            normalized = np.clip((dose_grid - q01) / (q99 - q01 + EPSILON), 0, 1)
            self.dose_data['normalization'] = {'method': 'robust', 'q01': q01, 'q99': q99}
            
        elif method == 'minmax':
            min_val = np.min(dose_grid)
            max_val = np.max(dose_grid)
            normalized = (dose_grid - min_val) / (max_val - min_val + EPSILON)
            self.dose_data['normalization'] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
        elif method == 'standard':
            mean_val = np.mean(dose_grid)
            std_val = np.std(dose_grid)
            normalized = (dose_grid - mean_val) / (std_val + EPSILON)
            self.dose_data['normalization'] = {'method': 'standard', 'mean': mean_val, 'std': std_val}
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.dose_data['dose_grid_normalized'] = normalized
        print(f"Applied {method} normalization")
        
        return normalized
    
    def denormalize_data(self, normalized_data):
        """Denormalize data back to original scale"""
        if 'normalization' not in self.dose_data:
            raise ValueError("No normalization information found.")
        
        norm_info = self.dose_data['normalization']
        method = norm_info['method']
        
        if method == 'robust':
            return normalized_data * (norm_info['q99'] - norm_info['q01']) + norm_info['q01']
        elif method == 'minmax':
            return normalized_data * (norm_info['max'] - norm_info['min']) + norm_info['min']
        elif method == 'standard':
            return normalized_data * norm_info['std'] + norm_info['mean']
        
        return normalized_data

class DataLoader:
    """Enhanced data loading and preprocessing utilities"""
    
    @staticmethod
    def load_dose_from_dict(data_dict: Dict, space_dims=None, world_bounds=None):
        """
        Load dose data from dictionary format - convenience function
        
        Args:
            data_dict: Dictionary where keys are z-coordinates and values are 2D data
            space_dims: Physical dimensions [x, y, z] in meters
            world_bounds: Physical bounds
            
        Returns:
            dict: Standardized dose_data format
        """
        processor = RadiationDataProcessor(space_dims, world_bounds)
        return processor.load_from_dict(data_dict, space_dims, world_bounds)
    
    @staticmethod
    def load_dose_from_numpy(dose_array, space_dims, world_bounds=None):
        """
        Load dose data from numpy array - convenience function
        
        Args:
            dose_array: 3D numpy array
            space_dims: Physical dimensions
            world_bounds: Physical bounds
            
        Returns:
            dict: Standardized dose_data format
        """
        processor = RadiationDataProcessor(space_dims, world_bounds)
        return processor.load_from_numpy(dose_array, space_dims, world_bounds)
    
    @staticmethod
    def sample_training_points(dose_data, num_samples=300, sampling_strategy='positive_weighted'):
        """
        Sample training points from dose data
        Enhanced version with more sampling strategies
        
        Args:
            dose_data: Dict from load_dose_* functions
            num_samples: Number of training points to sample
            sampling_strategy: 'positive_weighted', 'uniform', 'high_dose', 'positive_only', 'gradient_based'
        
        Returns:
            tuple: (sampled_points_xyz, sampled_doses_values, sampled_log_doses_values)
        """
        dose_grid = dose_data['dose_grid']
        world_min = dose_data['world_min']
        voxel_size = dose_data['voxel_size']
        grid_shape = dose_data['grid_shape']
        
        if sampling_strategy == 'positive_only':
            # Only sample from voxels with positive dose
            positive_indices = np.argwhere(dose_grid > EPSILON)
            if len(positive_indices) == 0:
                raise ValueError("No positive dose values found in data")
            
            if len(positive_indices) < num_samples:
                print(f"Warning: Only {len(positive_indices)} positive dose points available, using all")
                sampled_indices = positive_indices
            else:
                sample_idx = np.random.choice(len(positive_indices), size=num_samples, replace=False)
                sampled_indices = positive_indices[sample_idx]
                
        elif sampling_strategy == 'high_dose':
            # Sample from highest dose regions
            flat_dose = dose_grid.flatten()
            sorted_idx = np.argsort(flat_dose)[::-1]  # Descending order
            top_indices = sorted_idx[:num_samples]
            sampled_indices = np.array(np.unravel_index(top_indices, dose_grid.shape)).T
            
        elif sampling_strategy == 'uniform':
            # Uniform random sampling from all voxels
            total_voxels = np.prod(grid_shape)
            flat_indices = np.random.choice(total_voxels, size=num_samples, replace=False)
            sampled_indices = np.array(np.unravel_index(flat_indices, dose_grid.shape)).T
            
        elif sampling_strategy == 'gradient_based':
            # Sample from high gradient regions (similar to tool.py approach)
            positive_indices = np.argwhere(dose_grid > EPSILON)
            if len(positive_indices) == 0:
                raise ValueError("No positive dose values found in data")
            
            # Compute gradients
            try:
                grad_x = np.abs(np.diff(dose_grid, axis=0))
                grad_y = np.abs(np.diff(dose_grid, axis=1))
                grad_z = np.abs(np.diff(dose_grid, axis=2))
                
                # Create gradient magnitude map
                gradient_map = np.zeros_like(dose_grid)
                gradient_map[:-1] += grad_x
                gradient_map[:, :-1] += grad_y
                gradient_map[:, :, :-1] += grad_z
                
                # Combine random and gradient-based sampling
                random_points = int(num_samples * 0.3)
                gradient_points = num_samples - random_points
                
                # Random sampling from positive regions
                if len(positive_indices) >= random_points:
                    random_sample_idx = np.random.choice(len(positive_indices), size=random_points, replace=False)
                    random_sampled = positive_indices[random_sample_idx]
                else:
                    random_sampled = positive_indices
                
                # Gradient-based sampling
                flat_grad = gradient_map.flatten()
                grad_indices = np.argsort(flat_grad)[::-1]  # Descending order
                top_grad_indices = grad_indices[:min(gradient_points * 3, len(grad_indices))]
                
                if len(top_grad_indices) >= gradient_points:
                    selected_grad_idx = np.random.choice(top_grad_indices, size=gradient_points, replace=False)
                else:
                    selected_grad_idx = top_grad_indices
                
                grad_sampled = np.array(np.unravel_index(selected_grad_idx, dose_grid.shape)).T
                
                # Combine samples
                sampled_indices = np.vstack([random_sampled, grad_sampled])
                
            except Exception as e:
                print(f"Gradient sampling failed, using positive_only: {e}")
                # Fallback to positive_only
                if len(positive_indices) >= num_samples:
                    sample_idx = np.random.choice(len(positive_indices), size=num_samples, replace=False)
                    sampled_indices = positive_indices[sample_idx]
                else:
                    sampled_indices = positive_indices
                    
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Convert grid indices to world coordinates
        sampled_points_xyz = world_min + sampled_indices * voxel_size + voxel_size / 2.0
        sampled_doses_values = dose_grid[tuple(sampled_indices.T)]
        sampled_log_doses_values = np.log(sampled_doses_values + EPSILON)
        
        print(f"Sampled {len(sampled_indices)} training points using '{sampling_strategy}' strategy")
        print(f"Dose range in samples: {np.min(sampled_doses_values):.2e} to {np.max(sampled_doses_values):.2e}")
        
        return sampled_points_xyz, sampled_doses_values, sampled_log_doses_values 