#!/usr/bin/env python3
"""
采样方式对比测试脚本

此脚本用于对比三种采样方式的一致性:
1. Kriging/dataAnalysis.py 的原生采样方式
2. ComposeProject 的统一采样方式 (unified_sampling.py)
3. PINN/data_processing.py 的 Kriging 风格采样

使用方法:
    cd ComposeProject
    python compare_sampling_methods.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- 路径设置 ---
try:
    project_root = Path(__file__).parent.parent.resolve()
except NameError:
    project_root = Path('.').parent.resolve()

# 添加必要的模块路径
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'ComposeProject'))
sys.path.insert(0, str(project_root / 'ComposeProject' / 'src'))
sys.path.insert(0, str(project_root / 'ComposeProject' / 'src' / 'data'))
sys.path.insert(0, str(project_root / 'PINN'))
sys.path.insert(0, str(project_root / 'Kriging'))


def compare_sampling_methods():
    """对比三种采样方式"""
    
    print("=" * 80)
    print(" 采样方式对比测试 ".center(80))
    print("=" * 80)
    
    # ============== 1. 加载数据 ==============
    print("\n--- [步骤 1: 加载数据] ---")
    
    try:
        from dataAnalysis import get_data
        from PINN.data_processing import DataLoader, RadiationDataProcessor
        
        # 加载Kriging格式数据
        data_file_path = project_root / 'PINN' / 'DATA.xlsx'
        print(f"数据文件: {data_file_path}")
        
        kriging_data = get_data(str(data_file_path))
        print(f"✅ Kriging格式数据加载成功，共 {len(kriging_data)} 个Z层")
        
        # 加载PINN格式数据
        excel_data = pd.read_excel(data_file_path, sheet_name=None)
        if 'Sheet1' in excel_data:
            del excel_data['Sheet1']
        raw_data_dict = {int(k.split('_')[-1]): v for k, v in excel_data.items()}
        
        processor = RadiationDataProcessor()
        dose_data = processor.load_from_dict(raw_data_dict, space_dims=[20.0, 10.0, 10.0])
        grid_shape = dose_data['grid_shape']
        print(f"✅ PINN格式数据加载成功，网格形状: {grid_shape}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============== 2. 定义统一的采样参数 ==============
    print("\n--- [步骤 2: 设置采样参数] ---")
    
    # 根据实际数据形状调整采样参数
    # 数据形状: [137, 111, 72]，确保采样范围不超出边界
    sampling_params = {
        'box_origin': [5, 5, 5],
        'box_extent': [min(90, int(grid_shape[0]) - 10), 
                      min(90, int(grid_shape[1]) - 10), 
                      min(60, int(grid_shape[2]) - 10)],  # 调整为 [90, 90, 60]
        'step_sizes': [5],
        'x_y_reverse': True,
        'direction': '3vector'
    }
    print(f"采样参数: {sampling_params}")
    print(f"数据网格形状: {grid_shape}")
    
    # ============== 3. 方法1: Kriging原生采样 ==============
    print("\n--- [方法 1: Kriging/dataAnalysis.py 原生采样] ---")
    
    try:
        from dataAnalysis import training_sampling
        
        df_kriging = training_sampling(
            kriging_data, 
            center_x=48, center_y=45, center_z=5,
            inner_radius_max=30,
            step_sizes=sampling_params['step_sizes'],
            x_y_reverse=sampling_params['x_y_reverse'],
            use_box_area=True,
            use_box=[sampling_params['box_origin'], 
                    sampling_params['box_extent'][0],
                    sampling_params['box_extent'][1],
                    sampling_params['box_extent'][2]],
            direction=sampling_params['direction'],
            sourcepos=[]
        )
        df_kriging.drop_duplicates(inplace=True)
        df_kriging.reset_index(drop=True, inplace=True)
        
        print(f"✅ Kriging原生采样完成: {len(df_kriging)} 个点")
        print(f"   坐标范围: X=[{df_kriging['x'].min()}, {df_kriging['x'].max()}], "
              f"Y=[{df_kriging['y'].min()}, {df_kriging['y'].max()}], "
              f"Z=[{df_kriging['z'].min()}, {df_kriging['z'].max()}]")
        
    except Exception as e:
        print(f"❌ Kriging原生采样失败: {e}")
        import traceback
        traceback.print_exc()
        df_kriging = None
    
    # ============== 4. 方法2: ComposeProject统一采样 ==============
    print("\n--- [方法 2: ComposeProject/src/data/unified_sampling.py 采样] ---")
    
    try:
        from unified_sampling import UnifiedSampler, SamplingConfig
        
        config = SamplingConfig(
            use_box_area=True,
            box_origin=sampling_params['box_origin'],
            box_extent=sampling_params['box_extent'],
            step_sizes=sampling_params['step_sizes'],
            x_y_reverse=sampling_params['x_y_reverse'],
            direction=sampling_params['direction'],
            source_positions=[]
        )
        
        sampler = UnifiedSampler(data=kriging_data, dose_data=dose_data)
        df_unified = sampler.training_sampling(config)
        
        print(f"✅ 统一采样模块完成: {len(df_unified)} 个点")
        print(f"   坐标范围: X=[{df_unified['x'].min()}, {df_unified['x'].max()}], "
              f"Y=[{df_unified['y'].min()}, {df_unified['y'].max()}], "
              f"Z=[{df_unified['z'].min()}, {df_unified['z'].max()}]")
        
    except Exception as e:
        print(f"❌ 统一采样模块失败: {e}")
        import traceback
        traceback.print_exc()
        df_unified = None
    
    # ============== 5. 方法3: PINN Kriging风格采样 ==============
    print("\n--- [方法 3: PINN/data_processing.py Kriging风格采样] ---")
    
    try:
        points_pinn, values_pinn, log_values_pinn = DataLoader.sample_kriging_style(
            dose_data,
            box_origin=sampling_params['box_origin'],
            box_extent=sampling_params['box_extent'],
            step_sizes=sampling_params['step_sizes'],
            source_positions=None,
            source_exclusion_radius=30.0,
            direction=sampling_params['direction']
        )
        
        print(f"✅ PINN Kriging风格采样完成: {len(points_pinn)} 个点")
        print(f"   物理坐标范围: X=[{points_pinn[:, 0].min():.2f}, {points_pinn[:, 0].max():.2f}], "
              f"Y=[{points_pinn[:, 1].min():.2f}, {points_pinn[:, 1].max():.2f}], "
              f"Z=[{points_pinn[:, 2].min():.2f}, {points_pinn[:, 2].max():.2f}]")
        
    except Exception as e:
        print(f"❌ PINN Kriging风格采样失败: {e}")
        import traceback
        traceback.print_exc()
        points_pinn = None
    
    # ============== 6. 对比结果 ==============
    print("\n" + "=" * 80)
    print(" 对比结果 ".center(80, "="))
    print("=" * 80)
    
    # 对比采样点数量
    print("\n--- 采样点数量对比 ---")
    counts = {}
    if df_kriging is not None:
        counts['Kriging原生'] = len(df_kriging)
    if df_unified is not None:
        counts['统一采样'] = len(df_unified)
    if points_pinn is not None:
        counts['PINN Kriging风格'] = len(points_pinn)
    
    for name, count in counts.items():
        print(f"  {name}: {count} 个点")
    
    if len(set(counts.values())) == 1:
        print("  ✅ 所有方法采样点数量一致！")
    else:
        print("  ⚠️ 采样点数量存在差异")
    
    # 对比坐标一致性
    if df_kriging is not None and df_unified is not None:
        print("\n--- Kriging原生 vs 统一采样 坐标对比 ---")
        
        kriging_sorted = df_kriging.sort_values(['x', 'y', 'z']).reset_index(drop=True)
        unified_sorted = df_unified.sort_values(['x', 'y', 'z']).reset_index(drop=True)
        
        if len(kriging_sorted) == len(unified_sorted):
            coords_match = np.allclose(
                kriging_sorted[['x', 'y', 'z']].values,
                unified_sorted[['x', 'y', 'z']].values
            )
            values_match = np.allclose(
                kriging_sorted['target'].values,
                unified_sorted['target'].values,
                rtol=1e-5
            )
            
            if coords_match and values_match:
                print("  ✅ 坐标和值完全一致！")
            else:
                if not coords_match:
                    print("  ❌ 坐标不一致")
                if not values_match:
                    print("  ❌ 值不一致")
        else:
            print("  ⚠️ 点数不同，无法直接比较")
    
    # 显示样本数据
    print("\n--- 前5个采样点示例 ---")
    if df_kriging is not None:
        print("\nKriging原生 (网格索引):")
        print(df_kriging.head())
    
    if df_unified is not None:
        print("\n统一采样 (网格索引):")
        print(df_unified.head())
    
    if points_pinn is not None:
        print("\nPINN Kriging风格 (物理坐标):")
        print(pd.DataFrame({
            'x': points_pinn[:5, 0],
            'y': points_pinn[:5, 1],
            'z': points_pinn[:5, 2],
            'value': values_pinn[:5]
        }))
    
    print("\n" + "=" * 80)
    print(" 测试完成 ".center(80))
    print("=" * 80)
    
    return {
        'kriging': df_kriging,
        'unified': df_unified,
        'pinn': (points_pinn, values_pinn) if points_pinn is not None else None
    }


if __name__ == "__main__":
    compare_sampling_methods()
