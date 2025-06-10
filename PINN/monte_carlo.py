"""
GPU加速蒙特卡洛辐射输运模拟器
"""

import numpy as np
import cupy as cp
import time
from tqdm import tqdm

# 物理常数
MeV_to_JOULE = 1.60218e-13
EPSILON = 1e-30

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