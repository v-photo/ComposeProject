#!/usr/bin/env python3
"""
专门用于对比 "旧PINN子项目" 和 "新耦合项目" 训练数据处理流程的脚本。

本脚本将分别执行两个流程的数据加载、采样和转换步骤，
然后打印并比较最终生成的训练数据，以验证其一致性。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- 常量 ---
# 这个值需要和 pinn_core.py 以及 ComposeTools.py 中的值保持一致
EPSILON = 1e-30

def compare_data_pipelines():
    """执行数据处理流程对比"""

    # --- 路径设置 ---
    try:
        project_root = Path(__file__).parent.resolve()
    except NameError:
        project_root = Path('.').resolve()
    
    # 添加必要的模块路径
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'ComposeProject'))
    sys.path.insert(0, str(project_root / 'PINN'))
    # PINN_claude 的路径也需要，因为 ComposeTools 可能会间接依赖它
    sys.path.insert(0, str(project_root / 'PINN_claude'))

    print("=" * 80)
    print(" 正在对比两个项目的数据处理流程 ".center(80))
    print("=" * 80)

    try:
        # --- 流程1: 模拟旧/基准 PINN 子项目的数据处理 ---
        print("\n--- [流程1: (修正后) 使用与新项目相同的DataProcessor] ---")
        # 统一使用 ComposeTools 中从 PINN_claude 迁移过来的工具
        from ComposeTools import RadiationDataProcessor
        
        # 1.1 数据加载
        data_file_path = project_root / 'PINN' / 'DATA.xlsx'
        excel_data_old = pd.read_excel(data_file_path, sheet_name=None)
        if 'Sheet1' in excel_data_old:
            del excel_data_old['Sheet1']
        raw_data_dict_old = {int(k.split('_')[-1]): v for k, v in excel_data_old.items()}
        
        data_processor_old = RadiationDataProcessor()
        dose_data_old = data_processor_old.load_from_dict(raw_data_dict_old, space_dims=[20.0, 10.0, 10.0])
        print("✅ (旧) dose_data 对象创建完成。")

        # 1.2 采样
        np.random.seed(42)
        points_indices = np.random.choice(np.prod(dose_data_old['grid_shape']), 300, replace=False)
        points_indices_3d = np.array(np.unravel_index(points_indices, dose_data_old['grid_shape']))
        
        old_train_points = (dose_data_old['world_min'][:, np.newaxis] +
                            points_indices_3d * dose_data_old['voxel_size'][:, np.newaxis]).T
        old_train_values = dose_data_old['dose_grid'][tuple(points_indices_3d)]
        print("✅ (旧) 训练数据采样完成。")
        
        # 1.3 转换 (参照旧PINN项目的做法)
        old_train_points = old_train_points.astype(np.float32)
        old_train_log_values = np.log(old_train_values.astype(np.float32) + EPSILON)
        print("✅ (旧) 数据类型转换和对数变换完成。")
        
        
        # --- 流程2: 模拟新耦合项目的数据处理 ---
        print("\n--- [流程2: 新耦合项目] ---")
        # from ComposeTools import RadiationDataProcessor as NewDataProcessor
        # 导入语句已在上面统一，这里无需重复
        
        # 2.1 数据加载
        data_file_path_new = project_root / 'PINN' / 'DATA.xlsx' # 耦合测试脚本用的也是这个路径
        excel_data_new = pd.read_excel(data_file_path_new, sheet_name=None)
        if 'Sheet1' in excel_data_new:
            del excel_data_new['Sheet1']
        raw_data_dict_new = {int(k.split('_')[-1]): v for k, v in excel_data_new.items()}

        data_processor_new = RadiationDataProcessor()
        dose_data_new = data_processor_new.load_from_dict(raw_data_dict_new, space_dims=[20.0, 10.0, 10.0])
        print("✅ (新) dose_data 对象创建完成。")
        
        # 2.2 采样
        np.random.seed(42) # 使用相同的随机种子
        points_indices_new = np.random.choice(np.prod(dose_data_new['grid_shape']), 300, replace=False)
        points_indices_3d_new = np.array(np.unravel_index(points_indices_new, dose_data_new['grid_shape']))
        
        new_train_points = (dose_data_new['world_min'][:, np.newaxis] +
                            points_indices_3d_new * dose_data_new['voxel_size'][:, np.newaxis]).T
        new_train_values = dose_data_new['dose_grid'][tuple(points_indices_3d_new)]
        print("✅ (新) 训练数据采样完成。")
        
        # 2.3 转换 (参照 ComposeTools.py 中 adapter.fit 的做法)
        new_train_points = new_train_points.astype(np.float32)
        new_train_log_values = np.log(new_train_values.astype(np.float32) + EPSILON)
        print("✅ (新) 数据类型转换和对数变换完成。")

        # --- 对比结果 ---
        print("\n" + "=" * 80)
        print(" 对比结果 ".center(80, "="))
        print("=" * 80)
        
        # 比较 train_points
        print("\n--- 比较训练点坐标 (train_points) ---")
        print(f"(旧) 类型: {old_train_points.dtype}, 形状: {old_train_points.shape}")
        print(f"(新) 类型: {new_train_points.dtype}, 形状: {new_train_points.shape}")
        points_equal = np.array_equal(old_train_points, new_train_points)
        print(f"内容是否完全一致? {'✅ 是' if points_equal else '❌ 否'}")
        if not points_equal:
            diff = np.sum(np.abs(old_train_points - new_train_points))
            print(f"差异总和: {diff}")

        # 比较 train_log_values
        print("\n--- 比较对数变换后的剂量值 (train_log_values) ---")
        print(f"(旧) 类型: {old_train_log_values.dtype}, 形状: {old_train_log_values.shape}")
        print(f"(新) 类型: {new_train_log_values.dtype}, 形状: {new_train_log_values.shape}")
        values_equal = np.array_equal(old_train_log_values, new_train_log_values)
        print(f"内容是否完全一致? {'✅ 是' if values_equal else '❌ 否'}")
        if not values_equal:
            diff = np.sum(np.abs(old_train_log_values - new_train_log_values))
            print(f"差异总和: {diff}")
            
        print("\n--- 抽样打印部分数据 ---")
        print("前5个训练点 (旧):")
        print(old_train_points[:5])
        print("\n前5个训练点 (新):")
        print(new_train_points[:5])
        
        print("\n前5个对数剂量值 (旧):")
        print(old_train_log_values[:5])
        print("\n前5个对数剂量值 (新):")
        print(new_train_log_values[:5])

    except ImportError as e:
        print(f"\n❌ 错误：无法导入必要的模块。请检查路径和文件。")
        print(f"详细信息: {e}")
    except Exception as e:
        print(f"\n❌ 脚本执行时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_data_pipelines() 