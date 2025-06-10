#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
耦合项目 PINN 流程测试脚本 (V2)

本脚本用于通过 `PINNAdapterV2` 测试新的纯调用PINN工作流。
其数据加载、预处理和核心配置与基准脚本 (`PINN/PINNTest.py`) 和
V1测试脚本 (`ComposeProject/PINNTest.py`) 完全一致，
以进行公平、准确的对比。

此版本验证的是 `ComposeTools2.py` 中的适配器，它直接调用
`PINN` 子项目的功能，没有任何中间层或耦合逻辑。

运行方式 (在项目根目录下):
    python3 ComposeProject/PINNTest2.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# --- 路径设置 ---
# 将主项目目录添加到Python路径，确保能找到 `ComposeTools2` 和 `PINN`
try:
    # 定位到 /ComposeProject 目录
    current_dir = Path(__file__).parent.resolve()
    # 定位到项目根目录
    project_root = current_dir.parent
except NameError:
    # 在交互式环境中的回退方案
    project_root = Path('.').resolve()

# 确保两个关键路径都被添加
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / 'PINN') not in sys.path:
    # ComposeTools2 内部会添加PINN路径，但为保险起见这里也添加
    sys.path.insert(0, str(project_root / 'PINN'))


# --- 动态导入模块 (V2) ---
try:
    # 从 V2 工具集导入适配器和配置
    from ComposeProject.ComposeTools2 import PINNAdapterV2, ComposeConfigV2
    # 同样需要 RadiationDataProcessor 来准备 dose_data
    from tools import RadiationDataProcessor
    print("✅ (V2 Test) 模块导入成功 (PINNAdapterV2, ComposeConfigV2, RadiationDataProcessor)。")
except ImportError as e:
    print(f"❌ (V2 Test) 模块导入失败: {e}")
    print("请确保 `ComposeProject/ComposeTools2.py` 文件存在且无误。")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    """主执行函数"""
    print("\n" + " (V2) 开始执行纯调用PINN流程测试 ".center(80, "="))
    
    try:
        # --- 1. 数据加载和预处理 (与基准脚本完全相同) ---
        data_file_path = project_root / 'PINN' / 'DATA.xlsx'
        print(f"⏳ (V2 Test) 正在从 '{data_file_path}' 加载数据...")
        excel_data = pd.read_excel(data_file_path, sheet_name=None)
        
        # 移除Excel中的默认 'Sheet1'
        if 'Sheet1' in excel_data:
            del excel_data['Sheet1']
        
        # 将工作表名称（如 'Slice_1'）转换为整数键
        raw_data_dict = {int(k.split('_')[-1]): v for k, v in excel_data.items()}
        print("✅ (V2 Test) Excel数据加载和解析完成。")
        
        # 使用 PINN 子项目中的 RadiationDataProcessor
        data_processor = RadiationDataProcessor()
        dose_data = data_processor.load_from_dict(raw_data_dict, space_dims=[20.0, 10.0, 10.0])
        print("✅ (V2 Test) `dose_data` 对象创建完成。")

        # --- 2. 采样训练点 (与基准脚本完全相同) ---
        np.random.seed(42) # 固定随机种子以保证所有脚本采样一致
        num_samples = 300
        points_indices = np.random.choice(np.prod(dose_data['grid_shape']), num_samples, replace=False)
        points_indices_3d = np.array(np.unravel_index(points_indices, dose_data['grid_shape']))

        # 将体素索引转换为世界坐标
        train_points = (dose_data['world_min'][:, np.newaxis] +
                        points_indices_3d * dose_data['voxel_size'][:, np.newaxis]).T
                        
        # 获取对应的剂量值
        train_values = dose_data['dose_grid'][tuple(points_indices_3d)]
        print(f"✅ (V2 Test) {num_samples} 个训练数据采样完成。")
        
        # --- 3. 定义配置 (使用V2配置对象) ---
        pinn_config_v2 = ComposeConfigV2(
            epochs=1000,
            use_lbfgs=False,
            loss_weights=[1, 100],
            network_layers=[3, 32, 32, 32, 32, 1]
        )
        print("✅ (V2 Test) `ComposeConfigV2` 训练配置定义完成。")

        # --- 4. 运行训练 (通过AdapterV2) ---
        print("⏳ (V2 Test) 准备开始训练...")
        adapter_v2 = PINNAdapterV2(config=pinn_config_v2)
        
        # 核心调用：使用 adapter_v2.fit
        # 注意：这里我们直接传入配置中定义的参数，也可以通过kwargs覆盖
        adapter_v2.fit(
            X=train_points, 
            y=train_values,
            dose_data=dose_data # 传入完整的dose_data
        )

        # --- 5. (可选) 执行一次预测以验证 ---
        print("⏳ (V2 Test) 执行一次简单的预测以验证模型...")
        sample_prediction_points = train_points[:5]
        predictions = adapter_v2.predict(sample_prediction_points)
        print("✅ (V2 Test) 预测完成。样本预测结果:")
        for i in range(5):
            # 修复: 移除对 predictions[i] 的 [0] 索引，因为它可能是一维数组
            prediction_value = predictions[i] if predictions.ndim == 1 else predictions[i][0]
            print(f"  - 点 {sample_prediction_points[i]}: 真实值 = {train_values[i]:.4e}, 预测值 = {prediction_value:.4e}")

        print("\n" + "🎉 (V2) 纯调用PINN流程测试成功完成！ 🎉".center(80, "="))

    except Exception as e:
        print(f"\n❌ (V2 Test) 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 