#!/usr/bin/env python3
"""
PINN 子项目独立测试脚本 (基准)

本脚本用于独立运行 PINN 子项目的工作流，作为一个干净的、无耦合的基准。
它将执行数据加载、预处理、模型创建和训练。

运行方式 (在项目根目录下):
    python3 PINN/PINNTest.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# --- 路径设置 ---
# 将主项目目录添加到Python路径
try:
    # 定位到 /PINN 目录
    current_dir = Path(__file__).parent.resolve()
    # 定位到 /耦合项目 根目录
    project_root = current_dir.parent
except NameError:
    # 在交互式环境中的回退方案
    project_root = Path('.').resolve()
sys.path.insert(0, str(project_root))

# --- 动态导入模块 ---
try:
    from PINN.tools import PINNTrainer, RadiationDataProcessor, setup_deepxde_backend
    print("✅ (基准) 模块导入成功。")
except ImportError as e:
    print(f"❌ (基准) 模块导入失败: {e}")
    sys.exit(1)

# --- 全局设置 ---
setup_deepxde_backend()
torch.set_default_dtype(torch.float32)
print("✅ (基准) DeepXDE后端和PyTorch默认数据类型已设置。")


def main():
    """主执行函数"""
    print("\n" + " (基准) 开始执行独立PINN项目测试 ".center(80, "="))
    
    try:
        # --- 1. 数据加载和预处理 ---
        data_file_path = project_root / 'PINN' / 'DATA.xlsx'
        excel_data = pd.read_excel(data_file_path, sheet_name=None)
        
        if 'Sheet1' in excel_data:
            del excel_data['Sheet1']
        
        raw_data_dict = {int(k.split('_')[-1]): v for k, v in excel_data.items()}
        print("✅ (基准) 数据加载和预处理完成。")
        
        data_processor = RadiationDataProcessor()
        dose_data = data_processor.load_from_dict(raw_data_dict, space_dims=[20.0, 10.0, 10.0])
        print("✅ (基准) dose_data 对象创建完成。")

        # --- 2. 采样训练点 ---
        # 注意：在独立测试中，我们直接从 dose_data 中获取点，模拟最纯粹的流程
        # 为了与耦合项目测试对齐，我们假设有一组预定义的采样点
        # 这里我们手动创建一个简单的采样
        np.random.seed(42) # 固定随机种子以保证两个脚本采样一致
        points_indices = np.random.choice(np.prod(dose_data['grid_shape']), 300, replace=False)
        points_indices_3d = np.array(np.unravel_index(points_indices, dose_data['grid_shape']))
        
        # 修正广播错误：将 (3,) 形状的数组调整为 (3, 1) 以进行正确的元素乘法
        train_points = (dose_data['world_min'][:, np.newaxis] + 
                        points_indices_3d * dose_data['voxel_size'][:, np.newaxis]).T
                        
        train_values = dose_data['dose_grid'][tuple(points_indices_3d)]
        train_log_values = np.log(train_values + 1e-30)
        print("✅ (基准) 训练数据采样完成。")
        
        # --- 3. 定义配置 ---
        pinn_config = {
            'epochs': 1000, 'use_lbfgs': False, 'loss_weights': [1, 100],
            'network_config': {'layers': [3, 32, 32, 32, 32, 1], 'activation': 'tanh', 'initializer': 'Glorot uniform'},
            'physical_params': {'rho_material': 1.205, 'mass_energy_abs_coeff': 0.001901}
        }
        print("✅ (基准) 训练配置定义完成。")

        # --- 4. 运行训练 ---
        print("⏳ (基准) 准备开始训练...")
        trainer = PINNTrainer(physical_params=pinn_config['physical_params'])
        trainer.create_pinn_model(
            dose_data=dose_data,
            sampled_points_xyz=train_points,
            sampled_log_doses_values=train_log_values,
            network_config=pinn_config['network_config']
        )
        trainer.train(
            epochs=pinn_config['epochs'],
            use_lbfgs=pinn_config['use_lbfgs'],
            loss_weights=pinn_config['loss_weights'],
            display_every=200
        )
        
        print("\n" + "🎉 (基准) 独立PINN项目测试成功完成！ 🎉".center(80, "="))

    except Exception as e:
        print(f"\n❌ (基准) 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 