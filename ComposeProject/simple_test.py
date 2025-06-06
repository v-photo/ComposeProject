#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Block-Kriging × PINN 耦合重建工具 - 简单测试
使用真实的PINN/DATA.xlsx数据进行测试
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_data_loading():
    """测试数据加载功能"""
    print("=" * 60)
    print("测试: 真实数据加载")
    print("=" * 60)
    
    # 添加PINN路径
    pinn_dir = Path(__file__).parent.parent / "PINN"
    sys.path.insert(0, str(pinn_dir))
    
    try:
        from dataAnalysis import get_data
        from tools import DataLoader, setup_deepxde_backend
        
        # 设置DeepXDE后端
        setup_deepxde_backend()
        print("✅ PINN模块导入成功")
        
        # 加载DATA.xlsx数据
        data_file_path = pinn_dir / "DATA.xlsx"
        print(f"数据文件路径: {data_file_path}")
        
        data_dict = get_data(str(data_file_path))
        print(f"✅ 成功加载数据，包含 {len(data_dict)} 个z层")
        
        # 分析数据
        first_layer = data_dict[0]
        print(f"每层形状: {first_layer.shape}")
        
        # 使用DataLoader处理数据
        dose_data = DataLoader.load_dose_from_dict(
            data_dict=data_dict,
            space_dims=[20.0, 10.0, 10.0]
        )
        
        print(f"✅ 数据处理完成:")
        print(f"  - 剂量网格形状: {dose_data['grid_shape']}")
        print(f"  - 空间维度: {dose_data['space_dims']} m")
        
        # 采样训练数据
        train_points, train_values, _ = DataLoader.sample_training_points(
            dose_data, 
            num_samples=100,  # 使用较少的样本避免内存问题
            sampling_strategy='positive_only'
        )
        
        print(f"✅ 训练数据采样完成:")
        print(f"  - 训练点数: {len(train_points)}")
        print(f"  - 剂量值范围: {np.min(train_values):.4e} - {np.max(train_values):.4e}")
        
        return train_points, train_values, dose_data
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_kriging_basic(train_points, train_values):
    """测试基础Kriging功能"""
    print("\n" + "=" * 60)
    print("测试: 基础Kriging功能")
    print("=" * 60)
    
    try:
        from ComposeTools import KrigingAdapter, ComposeConfig
        
        # 配置
        config = ComposeConfig(gpu_enabled=True, verbose=True)
        kriging_adapter = KrigingAdapter(config)
        
        # 创建测试点
        test_points = train_points[:50] + np.random.normal(0, 0.1, (50, 3))
        
        print(f"开始Kriging测试:")
        print(f"  - 训练点数: {len(train_points)}")
        print(f"  - 测试点数: {len(test_points)}")
        
        # 训练
        start_time = time.time()
        kriging_adapter.fit(train_points, train_values)
        fit_time = time.time() - start_time
        
        # 预测
        start_time = time.time()
        predictions, variances = kriging_adapter.predict(test_points)
        pred_time = time.time() - start_time
        
        print(f"✅ Kriging测试完成:")
        print(f"  - 训练时间: {fit_time:.2f} 秒")
        print(f"  - 预测时间: {pred_time:.2f} 秒")
        print(f"  - 预测值范围: {np.min(predictions):.4e} - {np.max(predictions):.4e}")
        
        return predictions, variances
        
    except Exception as e:
        print(f"❌ Kriging测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_coupling_workflow():
    """测试耦合工作流程"""
    print("\n" + "=" * 60)
    print("测试: 耦合工作流程")
    print("=" * 60)
    
    try:
        # 加载数据
        result = test_data_loading()
        if result is None:
            print("❌ 数据加载失败，跳过工作流测试")
            return
            
        train_points, train_values, dose_data = result
        
        # 测试Kriging
        kriging_result = test_kriging_basic(train_points, train_values)
        if kriging_result is None:
            print("❌ Kriging测试失败")
            return
            
        print("✅ 所有基础测试通过!")
        
        # 尝试简化的方案1测试
        print("\n尝试简化的方案1测试...")
        from ComposeTools import CouplingWorkflow, ComposeConfig
        
        config = ComposeConfig(
            gpu_enabled=True,
            verbose=True,
            pinn_epochs=100,  # 减少训练轮数
            fusion_weight=0.6
        )
        
        workflow = CouplingWorkflow(config)
        
        # 创建简单的测试网格
        test_grid_shape = (10, 8, 8)  # 更小的网格
        x_test = np.linspace(dose_data['world_min'][0], dose_data['world_max'][0], test_grid_shape[0])
        y_test = np.linspace(dose_data['world_min'][1], dose_data['world_max'][1], test_grid_shape[1])
        z_test = np.linspace(dose_data['world_min'][2], dose_data['world_max'][2], test_grid_shape[2])
        
        X, Y, Z = np.meshgrid(x_test, y_test, z_test, indexing='ij')
        test_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        print(f"简化测试配置:")
        print(f"  - 训练点数: {len(train_points)}")
        print(f"  - 测试点数: {len(test_points)}")
        print(f"  - PINN训练轮数: {config.pinn_epochs}")
        
        # 运行简化的方案1
        results = workflow.run_mode1_pipeline(
            train_points=train_points,
            train_values=train_values,
            prediction_points=test_points,
            fusion_weight=config.fusion_weight,
            space_dims=dose_data['space_dims'].tolist(),
            world_bounds={'min': dose_data['world_min'], 'max': dose_data['world_max']},
            kriging_params={'variogram_model': 'linear'},
            epochs=config.pinn_epochs
        )
        
        if results:
            print("✅ 方案1简化测试成功!")
            print(f"  - PINN预测范围: {np.min(results['pinn_predictions']):.4e} - {np.max(results['pinn_predictions']):.4e}")
            print(f"  - 最终预测范围: {np.min(results['final_predictions']):.4e} - {np.max(results['final_predictions']):.4e}")
        
    except Exception as e:
        print(f"❌ 耦合工作流测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🚀 开始GPU Block-Kriging × PINN 耦合重建工具测试")
    print("使用真实的PINN/DATA.xlsx数据")
    
    # 运行所有测试
    test_coupling_workflow()
    
    print("\n" + "="*60)
    print("🎉 测试完成!")
    print("="*60)

if __name__ == "__main__":
    main() 