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
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 强制使用CPU进行调试
print("⚠️  [调试] 正在强制使用CPU模式运行...")
os.environ["DDE_BACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
sys.path.insert(0, str(project_root.parent / 'PINN_claude'))


# --- 动态导入模块 ---
try:
    # 统一使用 ComposeTools 中从 PINN_claude 迁移过来的工具
    # 绕过Adapter，直接导入其核心组件进行测试
    from ComposeTools import RadiationDataProcessor
    from pinn_core import PINNTrainer, setup_deepxde_backend, EPSILON
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
        data_file_path = project_root.parent / 'PINN_claude' / 'DATA.xlsx'
        excel_data = pd.read_excel(data_file_path, sheet_name=None)
        
        if 'Sheet1' in excel_data:
            del excel_data['Sheet1']
        
        raw_data_dict = {int(k.split('_')[-1]): v for k, v in excel_data.items()}
        print("✅ (耦合) 数据加载和预处理完成。")
        
        # 确保使用的是从 ComposeTools 导入的 Data Processor
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
        
        # 修正：确保训练点的数据类型是float32，以匹配模型期望
        train_points = train_points.astype(np.float32)
        
        print("✅ (耦合) 训练数据采样完成。")
        
        # --- 3. 定义配置 (与基准脚本完全相同) ---
        pinn_config = {
            'epochs': 1000, 'use_lbfgs': False, 'loss_weights': [1, 100],
            'network_config': {'layers': [3, 32, 32, 32, 32, 1], 'activation': 'tanh', 'initializer': 'Glorot uniform'},
        }
        print("✅ (耦合) 训练配置定义完成。")

        # --- 4. 运行训练 (绕过Adapter，直接使用PINNTrainer进行调试) ---
        print("⏳ (耦合) 准备开始训练 (直接调用PINNTrainer)...")

        # 4.1 手动执行数据预处理 (原在Adapter中完成)
        train_log_values = np.log(train_values.astype(np.float32) + EPSILON)

        # 4.2 创建并使用PINNTrainer
        trainer = PINNTrainer()

        # 4.3 创建模型
        print("--> 正在创建PINN模型...")
        trainer.create_pinn_model(
            dose_data=dose_data,
            sampled_points_xyz=train_points.astype(np.float32),
            sampled_log_doses_values=train_log_values,
            network_config=pinn_config['network_config'],
            include_source=False
        )
        
        # 4.4 训练模型
        print("--> 正在启动模型训练...")
        trainer.train(
            epochs=pinn_config['epochs'],
            loss_weights=pinn_config['loss_weights'],
            use_lbfgs=pinn_config['use_lbfgs'],
            display_every=200
        )

        print("\n" + "🎉 (耦合) 耦合项目PINN流程测试成功完成！ 🎉".center(80, "="))

    except Exception as e:
        print(f"\n❌ (耦合) 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 