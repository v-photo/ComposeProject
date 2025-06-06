"""
Physics-Informed Neural Networks (PINN) for radiation dose simulation
Tools and utility functions - Enhanced with flexible data import capabilities
"""

import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import deepxde as dde
import time
import torch
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from typing import Dict, Union, Tuple, List

# Physical constants
MeV_to_JOULE = 1.60218e-13
EPSILON = 1e-30

class SimulationConfig:
    """Configuration class for simulation parameters"""
    
    def __init__(self, 
                 space_dims=[20.0, 10.0, 10.0],
                 grid_shape=[100, 100, 100],
                 pinn_grid_shape=None,
                 source_energy_MeV=30.0,
                 n_particles=1000000,
                 rho_air_kg_m3=1.205,
                 mass_energy_abs_coeff=0.001901,
                 energy_cutoff_MeV=0.01):
        
        self.SPACE_DIMS_np = np.array(space_dims)
        self.GRID_SHAPE_np = np.array(grid_shape)
        self.PINN_GRID_SHAPE_np = np.array(pinn_grid_shape) if pinn_grid_shape is not None else np.array(grid_shape)
        self.SOURCE_ENERGY_MeV = source_energy_MeV
        self.N_PARTICLES = n_particles
        self.RHO_AIR_kg_m3 = rho_air_kg_m3
        self.MASS_ENERGY_ABS_COEFF_m2_kg = mass_energy_abs_coeff
        self.ENERGY_CUTOFF_MeV = energy_cutoff_MeV
        
        # Derived parameters
        self.WORLD_MIN_np = -self.SPACE_DIMS_np / 2.0
        self.WORLD_MAX_np = self.SPACE_DIMS_np / 2.0
        self.VOXEL_SIZE_np = self.SPACE_DIMS_np / self.GRID_SHAPE_np
        self.VOXEL_VOLUME_m3 = np.prod(self.VOXEL_SIZE_np)
        self.VOXEL_MASS_kg = self.RHO_AIR_kg_m3 * self.VOXEL_VOLUME_m3
        
        # GPU constants (optional, only if using internal MC)
        self._gpu_constants_initialized = False
    
    def setup_gpu_constants(self):
        """Setup GPU constants using CuPy (only when needed for internal MC)"""
        if self._gpu_constants_initialized:
            return
            
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
        dose_grid = np.zeros((x_size, y_size, z_size), dtype=np.float32)
        
        for z_idx, z_coord in enumerate(z_coords):
            layer_data = data_dict[z_coord]
            
            # Convert to numpy if needed
            if hasattr(layer_data, 'values'):
                layer_array = layer_data.values
            else:
                layer_array = np.array(layer_data)
            
            # Note: transpose to match expected grid convention (x, y, z)
            dose_grid[:, :, z_idx] = layer_array.T.astype(np.float32)
        
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
            'world_min': world_min.astype(np.float32),
            'world_max': world_max.astype(np.float32),
            'voxel_size': voxel_size.astype(np.float32),
            'grid_shape': grid_shape,
            'space_dims': self.space_dims.astype(np.float32),
            'z_coords': np.array(z_coords).astype(np.float32),
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
            'dose_grid': dose_array.astype(np.float32),
            'world_min': world_min.astype(np.float32),
            'world_max': world_max.astype(np.float32),
            'voxel_size': voxel_size.astype(np.float32),
            'grid_shape': grid_shape,
            'space_dims': self.space_dims.astype(np.float32)
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
        
        return sampled_points_xyz.astype(np.float32), sampled_doses_values.astype(np.float32), sampled_log_doses_values.astype(np.float32)

class GPUMonteCarloSimulator:
    """GPU-accelerated Monte Carlo radiation transport simulator (optional internal MC)"""
    
    def __init__(self, config):
        self.config = config
        # Initialize GPU constants only when needed
        self.config.setup_gpu_constants()
        self.energy_deposited_grid_gpu = cp.zeros(tuple(config.GRID_SHAPE_np), dtype=cp.float64)
    
    @staticmethod
    def get_random_isotropic_direction_gpu(num_particles):
        """Generate random isotropic directions for particles"""
        phi = 2 * cp.pi * cp.random.rand(num_particles, dtype=cp.float64)
        cos_theta = 2 * cp.random.rand(num_particles, dtype=cp.float64) - 1
        sin_theta = cp.sqrt(1 - cos_theta**2)
        dx = sin_theta * cp.cos(phi)
        dy = sin_theta * cp.sin(phi)
        dz = cos_theta
        return cp.stack((dx, dy, dz), axis=-1)
    
    @staticmethod
    def world_to_grid_index_gpu(pos_world_gpu, world_min_gpu, voxel_size_gpu, grid_shape_gpu):
        """Convert world coordinates to grid indices"""
        relative_pos = pos_world_gpu - world_min_gpu
        grid_idx_float = relative_pos / voxel_size_gpu
        grid_idx = cp.floor(grid_idx_float).astype(cp.int32)
        valid_mask = cp.all((grid_idx >= 0) & (grid_idx < grid_shape_gpu.reshape(1,3)), axis=1)
        return grid_idx, valid_mask
    
    @staticmethod
    def is_inside_geometry_gpu(pos_world_gpu, world_min_gpu, world_max_gpu):
        """Check if particles are inside the geometry"""
        return cp.all((pos_world_gpu >= world_min_gpu) & (pos_world_gpu < world_max_gpu), axis=1)
    
    @staticmethod
    def calculate_distance_to_voxel_boundary_gpu(pos_world_gpu, direction_gpu, current_voxel_idx_gpu, world_min_gpu, voxel_size_gpu):
        """Calculate distance to voxel boundary for ray tracing"""
        n_active = pos_world_gpu.shape[0]
        if n_active == 0:
            return cp.array([], dtype=cp.float64)
        
        voxel_min_world_active = world_min_gpu + current_voxel_idx_gpu * voxel_size_gpu
        voxel_max_world_active = voxel_min_world_active + voxel_size_gpu
        distances = cp.full((n_active, 3), cp.inf, dtype=cp.float64)
        
        for i in range(3):  # x, y, z
            mask_pos_dir = direction_gpu[:, i] > 1e-9
            if cp.any(mask_pos_dir):
                dist_to_b_pos = (voxel_max_world_active[mask_pos_dir, i] - pos_world_gpu[mask_pos_dir, i]) / direction_gpu[mask_pos_dir, i]
                distances[mask_pos_dir, i] = cp.where(dist_to_b_pos >= 0, dist_to_b_pos, cp.inf)
            
            mask_neg_dir = direction_gpu[:, i] < -1e-9
            if cp.any(mask_neg_dir):
                dist_to_b_neg = (voxel_min_world_active[mask_neg_dir, i] - pos_world_gpu[mask_neg_dir, i]) / direction_gpu[mask_neg_dir, i]
                distances[mask_neg_dir, i] = cp.where(dist_to_b_neg >= 0, dist_to_b_neg, cp.inf)
        
        min_dist = cp.min(distances, axis=1)
        min_dist = cp.where((min_dist == cp.inf) | (min_dist < 1e-9), 1e-6, min_dist)
        return min_dist + 1e-7
    
    def simulate(self, source_pos_world, max_simulation_steps=5000):
        """Run the Monte Carlo simulation (for internal MC usage)"""
        print(f"开始GPU蒙卡模拟 {self.config.N_PARTICLES} 个光子...")
        mc_start_time = time.time()
        
        # Initialize particles
        current_pos_world_all_gpu = cp.array(source_pos_world, dtype=cp.float64).reshape(1,3) + cp.zeros((self.config.N_PARTICLES, 3), dtype=cp.float64)
        current_energy_MeV_all_gpu = cp.full(self.config.N_PARTICLES, self.config.SOURCE_ENERGY_MeV_gpu, dtype=cp.float64)
        current_direction_all_gpu = self.get_random_isotropic_direction_gpu(self.config.N_PARTICLES)
        active_particles_mask = cp.ones(self.config.N_PARTICLES, dtype=bool)
        
        step_iter = 0
        
        with tqdm(total=self.config.N_PARTICLES, desc="GPU Simulating Particles") as pbar:
            last_sum_active = self.config.N_PARTICLES
            
            while cp.any(active_particles_mask) and step_iter < max_simulation_steps:
                step_iter += 1
                num_active_now = int(cp.sum(active_particles_mask))
                pbar.update(last_sum_active - num_active_now)
                last_sum_active = num_active_now
                
                if num_active_now == 0:
                    break
                
                # Get active particles
                pos_active = current_pos_world_all_gpu[active_particles_mask]
                energy_active = current_energy_MeV_all_gpu[active_particles_mask]
                dir_active = current_direction_all_gpu[active_particles_mask]
                
                # Check geometry bounds
                in_geometry_mask_subset = self.is_inside_geometry_gpu(pos_active, self.config.WORLD_MIN_gpu, self.config.WORLD_MAX_gpu)
                temp_full_geom_mask = cp.zeros(self.config.N_PARTICLES, dtype=bool)
                temp_full_geom_mask[active_particles_mask] = in_geometry_mask_subset
                active_particles_mask &= temp_full_geom_mask
                
                if not cp.any(active_particles_mask):
                    break
                
                # Update active particles
                pos_active = current_pos_world_all_gpu[active_particles_mask]
                energy_active = current_energy_MeV_all_gpu[active_particles_mask]
                dir_active = current_direction_all_gpu[active_particles_mask]
                
                # Get grid indices
                grid_idx_active, valid_grid_idx_mask_subset = self.world_to_grid_index_gpu(
                    pos_active, self.config.WORLD_MIN_gpu, self.config.VOXEL_SIZE_gpu, self.config.GRID_SHAPE_gpu
                )
                temp_full_grid_valid_mask = cp.zeros(self.config.N_PARTICLES, dtype=bool)
                temp_full_grid_valid_mask[active_particles_mask] = valid_grid_idx_mask_subset
                active_particles_mask &= temp_full_grid_valid_mask
                
                if not cp.any(active_particles_mask):
                    break
                
                # Final update
                pos_active = current_pos_world_all_gpu[active_particles_mask]
                energy_active = current_energy_MeV_all_gpu[active_particles_mask]
                dir_active = current_direction_all_gpu[active_particles_mask]
                grid_idx_active, _ = self.world_to_grid_index_gpu(
                    pos_active, self.config.WORLD_MIN_gpu, self.config.VOXEL_SIZE_gpu, self.config.GRID_SHAPE_gpu
                )
                
                # Calculate path length
                dist_to_b_active = self.calculate_distance_to_voxel_boundary_gpu(
                    pos_active, dir_active, grid_idx_active, self.config.WORLD_MIN_gpu, self.config.VOXEL_SIZE_gpu
                )
                path_length_active = cp.minimum(dist_to_b_active, cp.min(self.config.VOXEL_SIZE_gpu) / 2.0)
                path_length_active = cp.maximum(path_length_active, 1e-9)
                
                # Energy deposition
                energy_deposited_step_active = energy_active * \
                    (1 - cp.exp(-self.config.MASS_ENERGY_ABS_COEFF_m2_kg_gpu * self.config.RHO_AIR_kg_m3_gpu * path_length_active))
                
                ix, iy, iz = grid_idx_active[:, 0], grid_idx_active[:, 1], grid_idx_active[:, 2]
                cp.add.at(self.energy_deposited_grid_gpu, (ix, iy, iz), energy_deposited_step_active)
                
                # Update particle states
                current_energy_MeV_all_gpu[active_particles_mask] -= energy_deposited_step_active
                current_pos_world_all_gpu[active_particles_mask] += dir_active * path_length_active[:, cp.newaxis]
                
                # Remove low-energy particles
                low_energy_mask_subset = current_energy_MeV_all_gpu[active_particles_mask] < self.config.ENERGY_CUTOFF_MeV_gpu
                temp_full_low_energy_mask = cp.zeros(self.config.N_PARTICLES, dtype=bool)
                temp_full_low_energy_mask[active_particles_mask] = low_energy_mask_subset
                active_particles_mask[temp_full_low_energy_mask] = False
            
            pbar.update(last_sum_active - int(cp.sum(active_particles_mask)))
        
        mc_end_time = time.time()
        print(f"GPU蒙卡模拟完成. 耗时: {mc_end_time - mc_start_time:.2f} 秒")
        
        if step_iter >= max_simulation_steps:
            print(f"警告: 模拟在达到最大步数 {max_simulation_steps} 后停止。")
        
        # Convert to dose
        energy_deposited_grid_cpu = cp.asnumpy(self.energy_deposited_grid_gpu)
        dose_grid_Gy = (energy_deposited_grid_cpu * MeV_to_JOULE) / self.config.VOXEL_MASS_kg
        
        return dose_grid_Gy

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
        
        self.physical_params = physical_params
        self.k_initial_guess = physical_params.get('k_initial_guess', 
            physical_params['rho_material'] * physical_params['mass_energy_abs_coeff'])
        
        self.model = None
        self.k_pinn = None #important: 辐射衰减系数
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
        self.dose_data = dose_data
        self.world_min = dose_data['world_min']
        self.world_max = dose_data['world_max']
        self.geometry = dde.geometry.Cuboid(self.world_min, self.world_max)
        
        # Network configuration
        if network_config is None:
            if include_source:
                network_config = {'layers': [3] + [128] * 6 + [1], 'activation': 'tanh'}
            else:
                network_config = {'layers': [3] + [64] * 4 + [1], 'activation': 'tanh'}
        
        # Trainable parameters
        self.k_pinn = dde.Variable(float(self.k_initial_guess))
        
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
            
            def pde_with_source(x, u):
                return self._pde_with_source(x, u)
            
            pde_func = pde_with_source
        else:
            def pde_no_source(x, u):
                return self._pde_no_source(x, u)
            
            pde_func = pde_no_source
        
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
    
    def _pde_no_source(self, x, u):
        """PDE without source term"""
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_y = dde.grad.jacobian(u, x, i=0, j=1)
        du_z = dde.grad.jacobian(u, x, i=0, j=2)
        grad_u_sq = du_x**2 + du_y**2 + du_z**2
        
        du_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        du_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)
        laplacian_u = du_xx + du_yy + du_zz
        
        return grad_u_sq + laplacian_u - self.k_pinn**2
    
    def _pde_with_source(self, x, u):
        """PDE with source term"""
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_y = dde.grad.jacobian(u, x, i=0, j=1)
        du_z = dde.grad.jacobian(u, x, i=0, j=2)
        grad_u_sq = du_x**2 + du_y**2 + du_z**2
        
        du_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        du_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)
        laplacian_u = du_xx + du_yy + du_zz
        
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        
        sigma_s_sq = dde.backend.exp(self.source_params['log_sigma_s_sq'])
        sigma_s_sq_safe = dde.backend.relu(sigma_s_sq) + 1e-6
        
        source_term = self.source_params['As'] * dde.backend.exp(
            -((x_coord - self.source_params['xs'])**2 + 
              (y_coord - self.source_params['ys'])**2 + 
              (z_coord - self.source_params['zs'])**2) / sigma_s_sq_safe
        )
        
        return grad_u_sq + laplacian_u - self.k_pinn**2 - source_term
    
    def train(self, epochs=10000, use_lbfgs=True, loss_weights=None, display_every=500):
        """Train the PINN model with parameter monitoring"""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_pinn_model() first.")
            
        pinn_start_time = time.time()
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = [10, 100] if not self.source_params else [1, 10]
        
        # 收集所有可训练变量
        trainable_variables = [self.k_pinn]
        if self.source_params:
            trainable_variables.extend(list(self.source_params.values()))
        
        # Adam training
        self.model.compile("adam", lr=1e-3, loss_weights=loss_weights, 
                        external_trainable_variables=trainable_variables)
        
        # 分阶段训练，每个阶段后打印参数
        stages = [epochs // 4, epochs // 2, epochs * 3 // 4, epochs]
        last_epoch = 0
        
        for i, stage in enumerate(stages):
            current_epochs = stage - last_epoch
            if current_epochs > 0:
                losshistory, train_state = self.model.train(
                    iterations=current_epochs, 
                    display_every=100
                )
                
            last_epoch = stage
        
        # L-BFGS refinement
        if use_lbfgs:
            self.model.compile("L-BFGS", loss_weights=loss_weights, 
                            external_trainable_variables=trainable_variables)
            
            losshistory_lbfgs, train_state_lbfgs = self.model.train(
                display_every=50
            )
        
        pinn_end_time = time.time()
    
    def _get_scalar_value(self, variable):
        """Extract scalar value from DeepXDE variable"""
        val = variable.detach().cpu().numpy()
        return val.item() if val.ndim == 0 else val[0]
    
    def predict(self, prediction_points):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        log_dose_pinn = self.model.predict(prediction_points)
        dose_pinn = np.exp(log_dose_pinn[:, 0])
        dose_pinn[dose_pinn < EPSILON] = EPSILON
        
        return dose_pinn
    
    def get_learned_parameters(self):
        """Get the learned parameters"""
        k_val = self._get_scalar_value(self.k_pinn)
        
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
        
        return result

    def validate_parameter_updates(self, initial_params=None):
        """验证参数是否在训练过程中被更新"""
        if initial_params is None:
            return "请在训练前调用此方法并保存初始参数"
        
        current_k = self._get_scalar_value(self.k_pinn)
        initial_k = initial_params.get('k_pinn', self.k_initial_guess)
        
        if self.source_params and 'source_params' in initial_params:
            for name, var in self.source_params.items():
                current_val = self._get_scalar_value(var)
                initial_val = initial_params['source_params'][name]
                change = abs(current_val - initial_val)
        
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
        valid_comparison_mask = (dose_grid_mc < np.max(dose_grid_mc))
        diff_abs = np.abs(dose_pinn_on_mc_grid[valid_comparison_mask] - dose_grid_mc[valid_comparison_mask])
        relative_error = diff_abs / (np.abs(dose_grid_mc[valid_comparison_mask]) + EPSILON)
        mean_relative_error = np.mean(relative_error)
        
        mae = np.mean(np.abs(dose_pinn_on_mc_grid - dose_grid_mc))
        rmse = np.sqrt(np.mean((dose_pinn_on_mc_grid - dose_grid_mc)**2))
        
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
        
        return {
            'mean_relative_error': mean_relative_error,
            'mae': mae,
            'rmse': rmse,
            'dose_pinn_on_mc_grid': dose_pinn_on_mc_grid,
            'mc_grid_coords': (mc_x_centers, mc_y_centers, mc_z_centers),
            **training_results
        }

class Visualizer:
    """Visualization tools for PINN and Monte Carlo results"""
    
    @staticmethod
    def plot_comparison(dose_mc_data, dose_pinn_grid, pinn_grid_coords, evaluation_results,
                       source_positions=None, learned_params=None, 
                       num_particles=None, num_training_points=None):
        """
        Create comprehensive comparison plots
        
        Args:
            dose_mc_data: MC dose data dict from DataLoader
            dose_pinn_grid: PINN prediction grid
            pinn_grid_coords: PINN grid coordinates
            evaluation_results: Results from ResultAnalyzer
            source_positions: Dict with 'mc_source' and optionally 'learned_source'
            learned_params: Learned PINN parameters
            num_particles: Number of MC particles (for title)
            num_training_points: Number of training points (for title)
        """
        print("开始可视化...")
        
        dose_grid_mc = dose_mc_data['dose_grid']
        dose_pinn_on_mc_grid = evaluation_results['dose_pinn_on_mc_grid']
        mc_x_centers, mc_y_centers, mc_z_centers = evaluation_results['mc_grid_coords']
        
        # Common color scale
        common_vmin = min(np.min(dose_grid_mc[dose_grid_mc > EPSILON]),
                         np.min(dose_pinn_grid[dose_pinn_grid > EPSILON]))
        common_vmax = max(np.max(dose_grid_mc), np.max(dose_pinn_grid))
        if common_vmax <= common_vmin:
            common_vmin = EPSILON
            common_vmax = EPSILON * 10
        log_norm = LogNorm(vmin=common_vmin, vmax=common_vmax)
        
        # Slices for MC truth
        mc_grid_shape = dose_mc_data['grid_shape']
        slice_z_idx_mc = mc_grid_shape[2] // 2
        slice_y_idx_mc = mc_grid_shape[1] // 2
        slice_x_idx_mc = mc_grid_shape[0] // 2
        
        dose_slice_xy_mc = dose_grid_mc[:, :, slice_z_idx_mc].T
        dose_slice_xz_mc = dose_grid_mc[:, slice_y_idx_mc, :].T
        dose_slice_yz_mc = dose_grid_mc[slice_x_idx_mc, :, :].T
        
        # Slices for PINN prediction
        pinn_grid_shape = dose_pinn_grid.shape
        slice_z_idx_pinn = pinn_grid_shape[2] // 2
        slice_y_idx_pinn = pinn_grid_shape[1] // 2
        slice_x_idx_pinn = pinn_grid_shape[0] // 2
        
        dose_slice_xy_pinn = dose_pinn_grid[:, :, slice_z_idx_pinn].T
        dose_slice_xz_pinn = dose_pinn_grid[:, slice_y_idx_pinn, :].T
        dose_slice_yz_pinn = dose_pinn_grid[slice_x_idx_pinn, :, :].T
        
        # Grid extents
        world_min = dose_mc_data['world_min']
        world_max = dose_mc_data['world_max']
        
        pinn_x_coords, pinn_y_coords, pinn_z_coords = pinn_grid_coords
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Title
        title_parts = ["PINN vs MC Dose Comparison"]
        if num_particles:
            title_parts.append(f"{num_particles} MC particles")
        if num_training_points:
            title_parts.append(f"{num_training_points} PINN training points")
        
        title_str = " - ".join(title_parts)
        
        if learned_params and learned_params[1] is not None:  # With source parameters
            k_val, source_vals = learned_params
            title_str += f"\nk={k_val:.4f}, Learned Source: A={source_vals['As']:.2e}, "
            title_str += f"pos=({source_vals['xs']:.2f},{source_vals['ys']:.2f},{source_vals['zs']:.2f}), "
            title_str += f"sigma={source_vals['sigma_s']:.2f}"
        elif learned_params:  # Only k parameter
            k_val = learned_params[0] if isinstance(learned_params, tuple) else learned_params
            title_str += f"\nTrainable k: {k_val:.4f} (1/m)"
        
        # important
        # title_str += f"\nGlobal MRE (dose>1%max): {evaluation_results['mean_relative_error']:.2%}"
        title_str += f"\nGlobal MRE: {evaluation_results['mean_relative_error']:.2%}"
        
        fig.suptitle(title_str, fontsize=14)
        
        # XY Plane
        im_mc_xy = axes[0,0].imshow(dose_slice_xy_mc, origin='lower', aspect='auto', norm=log_norm,
                                   extent=[world_min[0], world_max[0], world_min[1], world_max[1]])
        axes[0,0].set_title(f'MC Truth: XY (Z ~ {mc_z_centers[slice_z_idx_mc]:.2f} m)')
        axes[0,0].set_xlabel('X (m)'); axes[0,0].set_ylabel('Y (m)')
        
        if source_positions and 'mc_source' in source_positions:
            mc_source = source_positions['mc_source']
            axes[0,0].plot(mc_source[0], mc_source[1], 'r+', markersize=10, label="MC Source")
            
        if source_positions and 'learned_source' in source_positions:
            learned_source = source_positions['learned_source']
            axes[0,0].plot(learned_source[0], learned_source[1], 'gx', markersize=10, label="PINN Source")
            axes[0,0].legend(fontsize='small')
        
        im_pinn_xy = axes[0,1].imshow(dose_slice_xy_pinn, origin='lower', aspect='auto', norm=log_norm,
                                     extent=[world_min[0], world_max[0], world_min[1], world_max[1]])
        axes[0,1].set_title(f'PINN: XY (Z ~ {pinn_z_coords[slice_z_idx_pinn]:.2f} m)')
        axes[0,1].set_xlabel('X (m)'); axes[0,1].set_ylabel('Y (m)')
        
        if source_positions and 'mc_source' in source_positions:
            axes[0,1].plot(mc_source[0], mc_source[1], 'r+', markersize=10)
        if source_positions and 'learned_source' in source_positions:
            axes[0,1].plot(learned_source[0], learned_source[1], 'gx', markersize=10)
        
        # XZ Plane
        im_mc_xz = axes[1,0].imshow(dose_slice_xz_mc, origin='lower', aspect='auto', norm=log_norm,
                                   extent=[world_min[0], world_max[0], world_min[2], world_max[2]])
        axes[1,0].set_title(f'MC Truth: XZ (Y ~ {mc_y_centers[slice_y_idx_mc]:.2f} m)')
        axes[1,0].set_xlabel('X (m)'); axes[1,0].set_ylabel('Z (m)')
        
        if source_positions and 'mc_source' in source_positions:
            axes[1,0].plot(mc_source[0], mc_source[2], 'r+', markersize=10)
        if source_positions and 'learned_source' in source_positions:
            axes[1,0].plot(learned_source[0], learned_source[2], 'gx', markersize=10)
        
        im_pinn_xz = axes[1,1].imshow(dose_slice_xz_pinn, origin='lower', aspect='auto', norm=log_norm,
                                     extent=[world_min[0], world_max[0], world_min[2], world_max[2]])
        axes[1,1].set_title(f'PINN: XZ (Y ~ {pinn_y_coords[slice_y_idx_pinn]:.2f} m)')
        axes[1,1].set_xlabel('X (m)'); axes[1,1].set_ylabel('Z (m)')
        
        if source_positions and 'mc_source' in source_positions:
            axes[1,1].plot(mc_source[0], mc_source[2], 'r+', markersize=10)
        if source_positions and 'learned_source' in source_positions:
            axes[1,1].plot(learned_source[0], learned_source[2], 'gx', markersize=10)
        
        # YZ Plane
        im_mc_yz = axes[2,0].imshow(dose_slice_yz_mc, origin='lower', aspect='auto', norm=log_norm,
                                   extent=[world_min[1], world_max[1], world_min[2], world_max[2]])
        axes[2,0].set_title(f'MC Truth: YZ (X ~ {mc_x_centers[slice_x_idx_mc]:.2f} m)')
        axes[2,0].set_xlabel('Y (m)'); axes[2,0].set_ylabel('Z (m)')
        
        if source_positions and 'mc_source' in source_positions:
            axes[2,0].plot(mc_source[1], mc_source[2], 'r+', markersize=10)
        if source_positions and 'learned_source' in source_positions:
            axes[2,0].plot(learned_source[1], learned_source[2], 'gx', markersize=10)
        
        im_pinn_yz = axes[2,1].imshow(dose_slice_yz_pinn, origin='lower', aspect='auto', norm=log_norm,
                                     extent=[world_min[1], world_max[1], world_min[2], world_max[2]])
        axes[2,1].set_title(f'PINN: YZ (X ~ {pinn_x_coords[slice_x_idx_pinn]:.2f} m)')
        axes[2,1].set_xlabel('Y (m)'); axes[2,1].set_ylabel('Z (m)')
        
        if source_positions and 'mc_source' in source_positions:
            axes[2,1].plot(mc_source[1], mc_source[2], 'r+', markersize=10)
        if source_positions and 'learned_source' in source_positions:
            axes[2,1].plot(learned_source[1], learned_source[2], 'gx', markersize=10)
        
        # Relative Difference Slices
        diff_slice_xy = (dose_pinn_on_mc_grid[:,:,slice_z_idx_mc].T - dose_slice_xy_mc) / (dose_slice_xy_mc + EPSILON)
        diff_slice_xz = (dose_pinn_on_mc_grid[:,slice_y_idx_mc,:].T - dose_slice_xz_mc) / (dose_slice_xz_mc + EPSILON)
        diff_slice_yz = (dose_pinn_on_mc_grid[slice_x_idx_mc,:,:].T - dose_slice_yz_mc) / (dose_slice_yz_mc + EPSILON)
        
        diff_vmax = 1.0
        diff_vmin = -1.0
        
        im_diff_xy = axes[0,2].imshow(diff_slice_xy, origin='lower', aspect='auto', cmap='coolwarm', 
                                     vmin=diff_vmin, vmax=diff_vmax,
                                     extent=[world_min[0], world_max[0], world_min[1], world_max[1]])
        axes[0,2].set_title(f'Rel Diff: XY (PINN-MC)/MC')
        fig.colorbar(im_diff_xy, ax=axes[0,2], label='(PINN-MC)/MC')
        
        im_diff_xz = axes[1,2].imshow(diff_slice_xz, origin='lower', aspect='auto', cmap='coolwarm', 
                                     vmin=diff_vmin, vmax=diff_vmax,
                                     extent=[world_min[0], world_max[0], world_min[2], world_max[2]])
        axes[1,2].set_title(f'Rel Diff: XZ (PINN-MC)/MC')
        fig.colorbar(im_diff_xz, ax=axes[1,2], label='(PINN-MC)/MC')
        
        im_diff_yz = axes[2,2].imshow(diff_slice_yz, origin='lower', aspect='auto', cmap='coolwarm', 
                                     vmin=diff_vmin, vmax=diff_vmax,
                                     extent=[world_min[1], world_max[1], world_min[2], world_max[2]])
        axes[2,2].set_title(f'Rel Diff: YZ (PINN-MC)/MC')
        fig.colorbar(im_diff_yz, ax=axes[2,2], label='(PINN-MC)/MC')
        
        fig.colorbar(im_mc_xy, ax=axes[:,0:2].ravel().tolist(), label='Dose (Gy/source particle)', shrink=0.6, aspect=20)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
    
    @staticmethod
    def plot_3d_comparison(dose_mc_data, dose_pinn_grid, source_positions=None, 
                          learned_params=None, num_samples=2000):
        """Create 3D scatter plot comparison"""
        dose_grid_mc = dose_mc_data['dose_grid']
        world_min = dose_mc_data['world_min']
        voxel_size = dose_mc_data['voxel_size']
        space_dims = dose_mc_data['space_dims']
        
        # Common color scale
        common_vmin = min(np.min(dose_grid_mc[dose_grid_mc > EPSILON]),
                         np.min(dose_pinn_grid[dose_pinn_grid > EPSILON]))
        common_vmax = max(np.max(dose_grid_mc), np.max(dose_pinn_grid))
        if common_vmax <= common_vmin:
            common_vmin = EPSILON
            common_vmax = EPSILON * 10
        log_norm = LogNorm(vmin=common_vmin, vmax=common_vmax)
        
        # MC Truth 3D Scatter
        active_voxels_mc = np.argwhere(dose_grid_mc > common_vmin)
        if len(active_voxels_mc) > num_samples:
            sample_idx_mc = np.random.choice(len(active_voxels_mc), num_samples, replace=False)
            active_voxels_mc = active_voxels_mc[sample_idx_mc]
        active_coords_mc = world_min + active_voxels_mc * voxel_size + voxel_size / 2.0
        active_doses_mc = dose_grid_mc[tuple(active_voxels_mc.T)]
        
        # PINN Prediction 3D Scatter
        active_voxels_pinn = np.argwhere(dose_pinn_grid > common_vmin)
        if len(active_voxels_pinn) > num_samples:
            sample_idx_pinn = np.random.choice(len(active_voxels_pinn), num_samples, replace=False)
            active_voxels_pinn = active_voxels_pinn[sample_idx_pinn]
        pinn_voxel_size = space_dims / np.array(dose_pinn_grid.shape)
        active_coords_pinn = world_min + active_voxels_pinn * pinn_voxel_size + pinn_voxel_size / 2.0
        active_doses_pinn = dose_pinn_grid[tuple(active_voxels_pinn.T)]
        
        fig_3d = plt.figure(figsize=(18, 8))
        ax_mc = fig_3d.add_subplot(121, projection='3d')
        sc_mc = ax_mc.scatter(active_coords_mc[:,0], active_coords_mc[:,1], active_coords_mc[:,2],
                             c=active_doses_mc, norm=log_norm, cmap='viridis', s=5)
        
        if source_positions and 'mc_source' in source_positions:
            mc_source = source_positions['mc_source']
            ax_mc.plot([mc_source[0]], [mc_source[1]], [mc_source[2]], 'r+', markersize=15, label="MC Source")
            
        if source_positions and 'learned_source' in source_positions:
            learned_source = source_positions['learned_source']
            ax_mc.plot([learned_source[0]], [learned_source[1]], [learned_source[2]], 'gx', markersize=15, mew=3, label="PINN Source")
            ax_mc.legend(fontsize='small')
        
        ax_mc.set_title('MC Truth 3D Dose (Sampled)')
        ax_mc.set_xlabel('X (m)'); ax_mc.set_ylabel('Y (m)'); ax_mc.set_zlabel('Z (m)')
        world_max = dose_mc_data['world_max']
        ax_mc.set_xlim(world_min[0], world_max[0])
        ax_mc.set_ylim(world_min[1], world_max[1])
        ax_mc.set_zlim(world_min[2], world_max[2])
        
        ax_pinn = fig_3d.add_subplot(122, projection='3d')
        sc_pinn = ax_pinn.scatter(active_coords_pinn[:,0], active_coords_pinn[:,1], active_coords_pinn[:,2],
                                 c=active_doses_pinn, norm=log_norm, cmap='viridis', s=5)
        
        if source_positions and 'mc_source' in source_positions:
            ax_pinn.plot([mc_source[0]], [mc_source[1]], [mc_source[2]], 'r+', markersize=15)
            
        if source_positions and 'learned_source' in source_positions:
            ax_pinn.plot([learned_source[0]], [learned_source[1]], [learned_source[2]], 'gx', markersize=15, mew=3)
        
        ax_pinn.set_title(f'PINN Predicted 3D Dose (Sampled)')
        ax_pinn.set_xlabel('X (m)'); ax_pinn.set_ylabel('Y (m)'); ax_pinn.set_zlabel('Z (m)')
        ax_pinn.set_xlim(world_min[0], world_max[0])
        ax_pinn.set_ylim(world_min[1], world_max[1])
        ax_pinn.set_zlim(world_min[2], world_max[2])
        
        fig_3d.colorbar(sc_mc, ax=[ax_mc, ax_pinn], label='Dose (Gy/source particle)', shrink=0.6)
        
        title_str = "3D Comparison: MC Truth vs PINN Prediction"
        if source_positions and 'mc_source' in source_positions:
            mc_source = source_positions['mc_source']
            title_str += f" (True Source at {mc_source} m)"
        if source_positions and 'learned_source' in source_positions:
            learned_source = source_positions['learned_source']
            title_str += f"\nLearned Source at ({learned_source[0]:.2f}, {learned_source[1]:.2f}, {learned_source[2]:.2f}) m"
        
        plt.suptitle(title_str, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

def setup_deepxde_backend():
    """Setup DeepXDE backend"""
    os.environ['DDE_BACKEND'] = 'pytorch'

# Example usage functions
def load_and_process_dict_data(data_dict, space_dims=None, world_bounds=None):
    """
    Example function showing how to load and process dictionary format data
    Compatible with tool.py RadiationDataset input format
    
    Args:
        data_dict: Dictionary {z: DataFrame[y, x]} or {z: numpy_array}
        space_dims: Physical dimensions [x, y, z] in meters
        world_bounds: Dict with 'min' and 'max' arrays
    
    Returns:
        RadiationDataProcessor: Configured processor with loaded data
    """
    print("Loading radiation data from dictionary format...")
    
    # Create processor
    processor = RadiationDataProcessor(space_dims, world_bounds)
    
    # Load data
    dose_data = processor.load_from_dict(data_dict, space_dims, world_bounds)
    
    print("Data loading complete!")
    print(f"Available keys in dose_data: {list(dose_data.keys())}")
    
    return processor
