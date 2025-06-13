#!/usr/bin/env python3
"""
耦合项目 PINNAdapter 工作流测试脚本

本脚本用于测试重构后的 PINNAdapter。
其所有参数（物理参数、训练参数、数据参数）均与 `run_pinn_benchmark` 函数保持一致，
以确保可以进行公平、准确的对比。

运行方式 (在项目根目录下):
    python3 ComposeProject/test_pinn_adapter.py
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- 路径设置 (确保能找到所有模块) ---
try:
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent
except NameError:
    project_root = Path('.').resolve()

# 将需要的模块路径添加到 sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'PINN'))
sys.path.insert(0, str(project_root / 'ComposeProject'))

# --- 动态导入模块 ---
try:
    from ComposeTools import PINNAdapter, ComposeConfig
    from PINN.pinn_core import ResultAnalyzer
    from PINN.visualization import Visualizer
    from PINN.tools import setup_deepxde_backend
    print("✅ 模块导入成功。")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# --- 全局设置 ---
setup_deepxde_backend()
print("✅ DeepXDE后端已设置。")


def test_pinn_adapter_workflow():
    """
    通过调用 PINNAdapter，完整地复现 `run_pinn_benchmark` 的流程。
    """
    print("\n" + "=" * 70)
    print(" 开始执行 PINNAdapter 工作流测试 ".center(70, "="))
    print("=" * 70)

    try:
        # --- 1. 定义配置 (与 run_pinn_benchmark 一致) ---
        compose_config = ComposeConfig(verbose=True)
        
        # 物理参数
        physical_params = {
            'rho_material': 1.205,
            'mass_energy_abs_coeff': 1.0
        }
        
        # 训练超参数
        train_params = {
            'epochs': 10000, # 与 benchmark 保持一致
            'use_lbfgs': True,
            'loss_weights': [1, 100],
            'network_config': {'layers': [3] + [32] * 4 + [1], 'activation': 'tanh'}
        }
        
        # 数据和采样参数
        data_params = {
            'data_path': os.path.join(project_root, 'PINN', 'DATA.xlsx'),
            'space_dims': [20.0, 10.0, 10.0],
            'num_samples': 300,
            'sampling_strategy': 'positive_only'
        }
        grid_shape = [136, 112, 72]  # 用于预测和评估

        # --- 2. 初始化并训练 Adapter ---
        print("🚀 步骤1: 初始化并训练 PINNAdapter...")
        adapter = PINNAdapter(physical_params=physical_params, config=compose_config)
        adapter.fit(
            data_path=data_params['data_path'],
            space_dims=data_params['space_dims'],
            num_samples=data_params['num_samples'],
            sampling_strategy=data_params['sampling_strategy'],
            **train_params
        )
        print("✅ PINNAdapter 训练完成!")

        # --- 3. 准备预测点和真实数据 ---
        print("\n🚀 步骤2: 准备全场预测点和真实数据...")
        dose_data = adapter.dose_data
        if dose_data is None:
            raise ValueError("训练后的 adapter.dose_data 为空，无法进行预测。")

        pred_x = dose_data['world_min'][0] + (np.arange(grid_shape[0]) + 0.5) * dose_data['voxel_size'][0]
        pred_y = dose_data['world_min'][1] + (np.arange(grid_shape[1]) + 0.5) * dose_data['voxel_size'][1]
        pred_z = dose_data['world_min'][2] + (np.arange(grid_shape[2]) + 0.5) * dose_data['voxel_size'][2]
        XX, YY, ZZ = np.meshgrid(pred_x, pred_y, pred_z, indexing='ij')
        prediction_points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
        print("✅ 预测点准备完毕。")

        # --- 4. 预测 ---
        print("\n🚀 步骤3: 使用训练好的 Adapter 进行全场预测...")
        predicted_doses = adapter.predict(prediction_points)
        predicted_doses_grid = predicted_doses.reshape(grid_shape)
        print("✅ 全场预测完成!")

        # --- 5. 评估与可视化 ---
        print("\n🚀 步骤4: 评估预测结果并可视化...")
        ground_truth_doses = dose_data['dose_grid']
        analyzer = ResultAnalyzer()
        evaluation_results = analyzer.evaluate_predictions(
            dose_pinn_grid=predicted_doses_grid,
            dose_mc_data=dose_data,
            pinn_grid_coords=(pred_x, pred_y, pred_z)
        )

        source_pos = dose_data['world_min'] + np.array(np.unravel_index(np.argmax(dose_data['dose_grid']), grid_shape)) * dose_data['voxel_size']
        slice_idx_z = np.argmin(np.abs(pred_z - source_pos[2]))

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        Visualizer.plot_slice(
            grid_coords=(pred_x, pred_y, pred_z), grid_data=predicted_doses_grid,
            slice_dim='z', slice_idx=slice_idx_z, title=f'PINN Adapter 预测 (R²={r2_score:.4f})'
        )
        plt.subplot(1, 2, 2)
        Visualizer.plot_slice(
            grid_coords=(pred_x, pred_y, pred_z), grid_data=ground_truth_doses,
            slice_dim='z', slice_idx=slice_idx_z, title='真实剂量'
        )
        plt.suptitle("PINNAdapter 工作流测试结果")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图像
        save_path = "pinn_adapter_test_result.png"
        plt.savefig(save_path)
        print(f"✅ 结果对比图已保存到: {save_path}")
        plt.show()

        print("\n" + "=" * 70)
        print("🎉 PINNAdapter 工作流测试成功完成! 🎉".center(70, "="))
        print("=" * 70)

    except Exception as e:
        import traceback
        print(f"\n❌ 测试执行失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_pinn_adapter_workflow() 