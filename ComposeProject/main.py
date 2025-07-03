#!/usr/bin/env python3
"""
PINN-Kriging 耦合系统主入口脚本
Main entry script for PINN-Kriging coupling system

用法示例：
1. 使用默认配置：python main.py
2. 使用预设配置：python main.py --preset kriging_only
3. 使用自定义配置文件：python main.py --config my_config.py
4. 快速测试：python main.py --preset quick_test
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import time

# 添加路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_environment():
    """设置运行环境"""
    # 解决WSL环境下的matplotlib显示问题
    import matplotlib
    matplotlib.use('Agg')
    
    # 设置随机种子
    np.random.seed(42)

def load_config(config_source=None, preset=None):
    """加载配置"""
    from config import Config, get_preset_config, default_config
    
    if preset:
        print(f"📋 使用预设配置: {preset}")
        config = get_preset_config(preset)
    elif config_source:
        print(f"📋 加载自定义配置文件: {config_source}")
        # 这里可以扩展支持从文件加载配置
        config = default_config
    else:
        print("📋 使用默认配置")
        config = default_config
    
    return config

def create_sample_data(config):
    """创建示例数据用于测试"""
    print("📊 创建示例数据...")
    
    # 创建模拟的dose_data
    space_dims = config.data.space_dims
    dose_data = {
        'world_min': np.array([0.0, 0.0, 0.0]),
        'world_max': np.array(space_dims),
        'space_dims': space_dims
    }
    
    # 创建模拟的训练数据
    np.random.seed(config.system.random_seed)
    num_samples = config.data.num_samples
    
    # 生成训练点
    train_points = np.random.rand(num_samples, 3) * space_dims
    
    # 生成模拟的剂量值（使用简单的函数）
    def simple_dose_function(x, y, z):
        return np.exp(-(x**2 + y**2 + z**2) / 100) * 1000
    
    train_values = simple_dose_function(train_points[:, 0], 
                                       train_points[:, 1], 
                                       train_points[:, 2])
    
    # 创建测试数据
    test_size = config.data.test_set_size
    test_points = np.random.rand(test_size, 3) * space_dims
    test_values = simple_dose_function(test_points[:, 0], 
                                      test_points[:, 1], 
                                      test_points[:, 2])
    test_data = np.hstack([test_points, test_values.reshape(-1, 1)])
    
    # 创建预测点
    pred_points = np.random.rand(1000, 3) * space_dims
    
    print(f"   ✅ 训练数据: {len(train_points)} 点")
    print(f"   ✅ 测试数据: {len(test_data)} 点") 
    print(f"   ✅ 预测点: {len(pred_points)} 点")
    
    return train_points, train_values, test_data, pred_points, dose_data

def run_coupling_workflow(config, train_points, train_values, test_data, pred_points, dose_data, method='auto'):
    """运行耦合工作流"""
    print("\n🚀 开始运行耦合工作流...")
    
    from ComposeTools import CouplingWorkflow
    
    # 创建工作流
    workflow = CouplingWorkflow(physical_params=config.pinn.physical_params)
    
    # 运行自动选择pipeline
    start_time = time.time()
    
    if method == 'auto':
        # 智能选择模式
        print("🤖 使用智能选择模式：自动选择最适合的预测方法")
        results = workflow.run_auto_selection_pipeline(
            train_points=train_points,
            train_values=train_values,
            prediction_points=pred_points,
            dose_data=dose_data,
            test_data=test_data,
            training_epochs=config.pinn.total_epochs // 4,
            num_collocation_points=config.pinn.num_collocation_points
        )
    elif method == 'kriging':
        # 强制使用Kriging
        print("⚙️ 强制使用Kriging方法")
        from ComposeTools import KrigingAdapter
        kriging_adapter = KrigingAdapter()
        kriging_adapter.fit(train_points, train_values)
        predictions = kriging_adapter.predict(pred_points)
        results = {
            'method_used': 'kriging',
            'final_predictions': predictions,
            'total_time': 0
        }
    elif method == 'pinn':
        # 强制使用PINN
        print("🧠 强制使用PINN方法")
        from ComposeTools import AdvancedPINNAdapter
        pinn_adapter = AdvancedPINNAdapter(config.pinn.physical_params)
        pinn_adapter.fit_from_memory(
            train_points=train_points,
            train_values=train_values,
            dose_data=dose_data,
            test_data=test_data,
            num_collocation_points=config.pinn.num_collocation_points
        )
        pinn_adapter.train_cycle(max_epochs=config.pinn.total_epochs // 4)
        predictions = pinn_adapter.predict(pred_points)
        results = {
            'method_used': 'pinn',
            'final_predictions': predictions,
            'pinn_adapter': pinn_adapter,
            'total_time': 0
        }
    else:
        raise ValueError(f"未知的方法: {method}。支持的方法: 'auto', 'kriging', 'pinn'")
    
    end_time = time.time()
    results['total_time'] = end_time - start_time
    
    print(f"\n✅ 工作流完成！")
    print(f"   - 使用方法: {results['method_used']}")
    print(f"   - 预测点数: {len(results['final_predictions'])}")
    print(f"   - 总耗时: {results['total_time']:.2f} 秒")
    
    return results

def analyze_results(results, test_data):
    """分析结果"""
    print("\n📊 结果分析...")
    
    predictions = results['final_predictions']
    method_used = results['method_used']
    
    # 如果有测试数据，计算误差指标
    if test_data is not None and len(test_data) > 0:
        # 这里需要确保预测点和测试点对应
        # 为了简化，我们只计算一些基本统计
        print(f"   - 预测值范围: [{np.min(predictions):.2e}, {np.max(predictions):.2e}]")
        print(f"   - 预测值均值: {np.mean(predictions):.2e}")
        print(f"   - 预测值标准差: {np.std(predictions):.2e}")
        
        if method_used == 'kriging':
            print("   - 使用了Kriging方法，适合均匀分布的数据")
        else:
            print("   - 使用了高级PINN方法，适合复杂分布的数据")
            
            # 如果有PINN适配器，显示训练历史
            if 'pinn_adapter' in results:
                adapter = results['pinn_adapter']
                if hasattr(adapter, 'mre_history') and len(adapter.mre_history) > 0:
                    final_mre = adapter.mre_history[-1]
                    print(f"   - 最终MRE: {final_mre:.6f}")
                    print(f"   - 训练历史长度: {len(adapter.mre_history)}")

def save_results(results, config):
    """保存结果"""
    if not config.system.save_results:
        return
        
    print("\n💾 保存结果...")
    
    results_dir = Path(config.system.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # 保存预测结果
    predictions_file = results_dir / f"predictions_{config.experiment.experiment_name}.npy"
    np.save(predictions_file, results['final_predictions'])
    print(f"   ✅ 预测结果已保存: {predictions_file}")
    
    # 如果有PINN适配器，保存训练历史
    if 'pinn_adapter' in results:
        adapter = results['pinn_adapter']
        if hasattr(adapter, 'mre_history'):
            history_file = results_dir / f"training_history_{config.experiment.experiment_name}.npz"
            np.savez(history_file, 
                    mre_history=adapter.mre_history,
                    epoch_history=adapter.epoch_history,
                    training_events=adapter.training_events)
            print(f"   ✅ 训练历史已保存: {history_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PINN-Kriging耦合系统")
    parser.add_argument('--preset', type=str, 
                       choices=['full_adaptive', 'kriging_only', 'data_injection_only', 'baseline', 'quick_test'],
                       help='使用预设配置')
    parser.add_argument('--config', type=str, help='自定义配置文件路径')
    parser.add_argument('--method', type=str, choices=['auto', 'kriging', 'pinn'], 
                       default='auto', help='预测方法选择: auto(智能选择), kriging(强制克里金), pinn(强制PINN)')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 加载配置
    config = load_config(config_source=args.config, preset=args.preset)
    
    if args.verbose:
        config.system.verbose = True
    
    # 显示配置摘要
    print(config.summary())
    
    try:
        # 创建示例数据
        train_points, train_values, test_data, pred_points, dose_data = create_sample_data(config)
        
        # 运行耦合工作流
        results = run_coupling_workflow(config, train_points, train_values, test_data, pred_points, dose_data, method=args.method)
        
        # 分析结果
        analyze_results(results, test_data)
        
        # 保存结果
        save_results(results, config)
        
        print("\n🎉 运行完成！")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()