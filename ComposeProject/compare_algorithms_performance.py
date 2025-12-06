#!/usr/bin/env python3
"""
PINN vs Kriging 算法性能对比脚本

此脚本使用统一的采样方式获取训练数据，然后分别训练 PINN 和 Kriging 模型，
在相同测试集上评估性能，并生成详细的对比报告。

使用方法:
    cd ComposeProject
    python compare_algorithms_performance.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Tuple

# --- 路径设置 ---
try:
    project_root = Path(__file__).parent.parent.resolve()
except NameError:
    project_root = Path('.').parent.resolve()

# 添加必要的模块路径
sys.path.insert(0, str(project_root / 'PINN'))
sys.path.insert(0, str(project_root / 'Kriging'))
sys.path.insert(0, str(project_root / 'ComposeProject'))
sys.path.insert(0, str(project_root / 'ComposeProject' / 'src'))

sys.path.insert(0, str(project_root/'Kriging'))


def load_data() -> Dict[str, Any]:
    """加载和处理数据"""
    print("=== 加载数据 ===")
    
    try:
        from dataAnalysis import get_data
        from data_processing import RadiationDataProcessor
        
        # 加载Kriging格式数据
        data_file_path = Path("../PINN/DATA.xlsx")
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
        
        return {
            'kriging_data': kriging_data,
            'dose_data': dose_data,
            'grid_shape': grid_shape
        }
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_training_data(dose_data: Dict, num_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """获取训练数据（使用统一的Kriging风格采样）"""
    print(f"\n=== 获取训练数据 ({num_samples} 个样本) ===")
    
    try:
        from src.data.loader import sample_kriging_style
        
        # 使用Kriging风格采样确保一致性
        train_points, train_values = sample_kriging_style(
            dose_data,
            box_origin=[5, 5, 5],
            box_extent=[90, 90, 60],
            step_sizes=[5],
            source_positions=[],  # 不排除源点
            source_exclusion_radius=0.0
        )
        
        print(f"✅ 获取到 {len(train_points)} 个训练样本")
        print(f"   坐标范围: X=[{train_points[:, 0].min():.2f}, {train_points[:, 0].max():.2f}]")
        print(f"   值范围: [{train_values.min():.2e}, {train_values.max():.2e}]")
        
        return train_points, train_values
        
    except Exception as e:
        print(f"❌ 获取训练数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_test_data(dose_data: Dict, num_test_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """创建测试数据集"""
    print(f"\n=== 创建测试数据集 ({num_test_points} 个测试点) ===")
    
    try:
        from src.data.loader import sample_kriging_style
        
        # 从不同区域采样测试点
        test_points, test_values = sample_kriging_style(
            dose_data,
            box_origin=[5, 5, 5],  # 不同的起始位置
            box_extent=[90, 90, 60],  # 不同的范围
            step_sizes=[3],           # 不同的步长
            source_positions=[],
            source_exclusion_radius=0.0
        )
        
        # 如果采样点太多，随机选择指定数量
        if len(test_points) > num_test_points:
            indices = np.random.choice(len(test_points), num_test_points, replace=False)
            test_points = test_points[indices]
            test_values = test_values[indices]
        
        print(f"✅ 创建测试集: {len(test_points)} 个点")
        print(f"   值范围: [{test_values.min():.2e}, {test_values.max():.2e}]")
        
        return test_points, test_values
        
    except Exception as e:
        print(f"❌ 创建测试集失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_pinn_model(train_points: np.ndarray, train_values: np.ndarray, 
                    dose_data: Dict) -> Tuple[Any, float]:
    """训练PINN模型"""
    print("\n=== 训练 PINN 模型 ===")
    
    try:
        from pinn_core import PINNTrainer
        
        start_time = time.time()
        
        physical_params = {
            'rho_material': 1.205,        # 空气密度 kg/m³
            'mass_energy_abs_coeff': 1.0  # 质能吸收系数
        }
        
        trainer = PINNTrainer(physical_params=physical_params)
        
        # 创建模型
        trainer.create_pinn_model(
            dose_data=dose_data,
            sampled_points_xyz=train_points,
            sampled_log_doses_values=np.log(train_values + 1e-10).flatten(),
            include_source=False,
            network_config={'layers': [3, 64, 64, 64, 1], 'activation': 'tanh'}
        )
        
        # 训练（使用较少的轮数以便快速对比）
        trainer.train(epochs=2000, use_lbfgs=False, loss_weights=[1, 100])
        
        training_time = time.time() - start_time
        print(f"✅ PINN训练完成，耗时: {training_time:.2f} 秒")
        
        return trainer, training_time
        
    except Exception as e:
        print(f"❌ PINN训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def train_kriging_model(train_points: np.ndarray, train_values: np.ndarray) -> Tuple[Any, float]:
    """训练Kriging模型"""
    print("\n=== 训练 Kriging 模型 ===")
    
    try:
        from myKriging import training
        
        start_time = time.time()
        
        # 准备训练数据
        train_df = pd.DataFrame({
            'x': train_points[:, 0],
            'y': train_points[:, 1],
            'z': train_points[:, 2],
            'target': train_values.flatten()
        })
        
        # 训练模型
        model = training(
            df=train_df,
            variogram_model="exponential",  # 变异函数模型
            nlags=20,                        # 距离分组数
            enable_plotting=False,          # 不显示绘图
            weight=True,                   # 不使用加权
            uk=False,                       # 普通Kriging
            cpu_on=False                    # 使用GPU
        )
        
        training_time = time.time() - start_time
        print(f"✅ Kriging训练完成，耗时: {training_time:.2f} 秒")
        
        return model, training_time
        
    except Exception as e:
        print(f"❌ Kriging训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def evaluate_model(model, model_type: str, test_points: np.ndarray, 
                  test_values: np.ndarray) -> Dict[str, float]:
    """评估模型性能"""
    print(f"\n=== 评估 {model_type} 模型性能 ===")
    
    try:
        if model_type == 'PINN':
            # PINN预测
            predictions = model.predict(test_points)
            predictions = predictions.flatten()
            
        elif model_type == 'Kriging':
            # Kriging预测
            from myKriging import testing
            
            test_df = pd.DataFrame({
                'x': test_points[:, 0],
                'y': test_points[:, 1],
                'z': test_points[:, 2],
                'target': np.zeros(len(test_points))  # 虚拟值
            })
            
            predictions, _ = testing(
                df=test_df,
                model=model,
                block_size=10000,
                cpu_on=False,
                style="gpu_b",
                compute_precision=False
            )
            predictions = predictions.flatten()
        
        # 计算评估指标
        from src.analysis.metrics import MetricsCalculator
        metrics = MetricsCalculator.compute_metrics(test_values, predictions)
        
        print(f"✅ {model_type}评估完成:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  - {metric_name}: {value:.6f}")
            else:
                print(f"  - {metric_name}: {value}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ {model_type}评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def compare_algorithms_performance():
    """主函数：对比PINN和Kriging算法性能"""
    
    print("=" * 80)
    print("  PINN vs Kriging 算法性能对比  ".center(80))
    print("=" * 80)
    
    # 1. 加载数据
    data = load_data()
    if data is None:
        return
    
    # 2. 获取训练数据
    train_points, train_values = get_training_data(data['dose_data'], num_samples=300)
    if train_points is None:
        return
    
    # 3. 创建测试数据
    test_points, test_values = create_test_data(data['dose_data'], num_test_points=500)
    if test_points is None:
        return
    
    # 4. 训练模型
    pinn_model, pinn_time = train_pinn_model(train_points, train_values, data['dose_data'])
    kriging_model, kriging_time = train_kriging_model(train_points, train_values)
    
    if pinn_model is None or kriging_model is None:
        print("❌ 模型训练失败，无法进行对比")
        return
    
    # 5. 评估模型
    pinn_metrics = evaluate_model(pinn_model, 'PINN', test_points, test_values)
    kriging_metrics = evaluate_model(kriging_model, 'Kriging', test_points, test_values)
    
    # 6. 生成对比报告
    print("\n" + "=" * 80)
    print("  性能对比报告  ".center(80, "="))
    print("=" * 80)
    
    print("\n📊 实验设置:")
    print(f"  - 训练样本数: {len(train_points)}")
    print(f"  - 测试样本数: {len(test_points)}")
    print(f"  - 采样方式: Kriging风格统一采样")
    print(f"  - PINN网络: [3, 64, 64, 64, 1]")
    print(f"  - Kriging变异函数: exponential")
    
    print("\n⏱️  训练时间:")
    print(f"  - PINN: {pinn_time:.2f} 秒")
    print(f"  - Kriging: {kriging_time:.2f} 秒")
    time_ratio = pinn_time / kriging_time if kriging_time > 0 else float('inf')
    print(f"  - 时间比: {time_ratio:.1f}x")
    
    print("\n🎯 性能对比 (测试集):")
    metrics_to_compare = ['MAE', 'RMSE', 'MAPE', 'R2']
    
    print(f"{'Metric':<8} {'PINN':>10} {'Kriging':>10} {'Winner':>10}")
    print("-" * 50)
    
    for metric in metrics_to_compare:
        pinn_val = pinn_metrics.get(metric, float('nan'))
        kriging_val = kriging_metrics.get(metric, float('nan'))
        
        if metric in ['MAE', 'RMSE', 'MAPE']:
            # 越小越好
            if pinn_val < kriging_val:
                winner = "🏆 PINN"
            else:
                winner = "🏆 Kriging"
        elif metric == 'R2':
            # 越大越好
            if pinn_val > kriging_val:
                winner = "🏆 PINN"
            else:
                winner = "🏆 Kriging"
        else:
            winner = "   -   "
        
        print(f"{metric:<8} {pinn_val:>10.4f} {kriging_val:>10.4f} {winner}")
    
    # 总结
    print("\n📋 总结:")
    print(f"  - 训练数据完全一致（{len(train_points)} 个点）")
    print(f"  - 测试数据完全一致（{len(test_points)} 个点）")
    print("  - 消除了采样差异，确保公平对比")
    
    # 保存结果（可选）
    try:
        results = {
            'experiment_info': {
                'train_samples': len(train_points),
                'test_samples': len(test_points),
                'sampling_method': 'kriging_style_unified',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'training_times': {
                'pinn': pinn_time,
                'kriging': kriging_time
            },
            'pinn_metrics': pinn_metrics,
            'kriging_metrics': kriging_metrics
        }
        
        import json
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / f'algorithm_comparison_{time.strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 结果已保存到: {results_file}")
        
    except Exception as e:
        print(f"\n⚠️  保存结果失败: {e}")
    
    print("\n🎉 算法性能对比完成！")
    print("=" * 80)


if __name__ == "__main__":
    compare_algorithms_performance()
