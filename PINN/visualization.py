"""
PINN和蒙特卡洛结果的可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# 常数
EPSILON = 1e-30

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