#!/usr/bin/env python3
"""
耦合项目 PINN 流程测试脚本

本脚本用于通过 PINNAdapter 测试耦合项目中的 PINN 工作流。
其数据加载、预处理和配置与基准脚本 (PINN/PINNTest.py) 完全一致，
以进行公平、准确的对比。

运行方式 (在项目根目录下):
    python3 ComposeProject/PINNTest.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# --- 路径设置 ---
# 将主项目目录添加到Python路径
try:
    # 定位到 /ComposeProject 目录
    current_dir = Path(__file__).parent.resolve()
    # 定位到 /耦合项目 根目录
    project_root = current_dir.parent
except NameError:
    # 在交互式环境中的回退方案
    project_root = Path('.').resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'ComposeProject'))
sys.path.insert(0, str(project_root / 'PINN'))


# --- 动态导入模块 ---
try:
    from ComposeTools import PINNAdapter, ComposeConfig
    # 同样需要 RadiationDataProcessor 来准备 dose_data
    from PINN.tools import RadiationDataProcessor, setup_deepxde_backend
    print("✅ (耦合) 模块导入成功。")
except ImportError as e:
    print(f"❌ (耦合) 模块导入失败: {e}")
    sys.exit(1)

# --- 全局设置 ---
setup_deepxde_backend()
torch.set_default_dtype(torch.float32)
print("✅ (耦合) DeepXDE后端和PyTorch默认数据类型已设置。")


def main():
    """主执行函数"""
    print("\n" + " (耦合) 开始执行耦合项目PINN流程测试 ".center(80, "="))
    
    try:
        # --- 1. 数据加载和预处理 (与基准脚本完全相同) ---
        data_file_path = project_root / 'PINN' / 'DATA.xlsx'
        excel_data = pd.read_excel(data_file_path, sheet_name=None)
        
        if 'Sheet1' in excel_data:
            del excel_data['Sheet1']
        
        raw_data_dict = {int(k.split('_')[-1]): v for k, v in excel_data.items()}
        print("✅ (耦合) 数据加载和预处理完成。")
        
        data_processor = RadiationDataProcessor()
        dose_data = data_processor.load_from_dict(raw_data_dict, space_dims=[20.0, 10.0, 10.0])
        print("✅ (耦合) dose_data 对象创建完成。")

        # --- 2. 采样训练点 (与基准脚本完全相同) ---
        np.random.seed(42) # 固定随机种子以保证两个脚本采样一致
        points_indices = np.random.choice(np.prod(dose_data['grid_shape']), 300, replace=False)
        points_indices_3d = np.array(np.unravel_index(points_indices, dose_data['grid_shape']))

        # 修正广播错误：将 (3,) 形状的数组调整为 (3, 1) 以进行正确的元素乘法
        train_points = (dose_data['world_min'][:, np.newaxis] +
                        points_indices_3d * dose_data['voxel_size'][:, np.newaxis]).T
                        
        train_values = dose_data['dose_grid'][tuple(points_indices_3d)]
        print("✅ (耦合) 训练数据采样完成。")
        
        # --- 3. 定义配置 (与基准脚本完全相同) ---
        pinn_config = {
            'epochs': 1000, 'use_lbfgs': False, 'loss_weights': [1, 100],
            'network_config': {'layers': [3, 32, 32, 32, 32, 1], 'activation': 'tanh', 'initializer': 'Glorot uniform'},
        }
        print("✅ (耦合) 训练配置定义完成。")

        # --- 4. 运行训练 (通过Adapter) ---
        print("⏳ (耦合) 准备开始训练...")
        adapter = PINNAdapter(config=ComposeConfig())
        
        # 核心调用：使用adapter.fit
        adapter.fit(
            X=train_points, 
            y=train_values,
            dose_data=dose_data, # 传入完整的dose_data
            epochs=pinn_config['epochs'],
            loss_weights=pinn_config['loss_weights'],
            use_lbfgs=pinn_config['use_lbfgs'],
            network_config=pinn_config['network_config']
        )
        
        print("\n" + "🎉 (耦合) 耦合项目PINN流程测试成功完成！ 🎉".center(80, "="))

    except Exception as e:
        print(f"\n❌ (耦合) 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 