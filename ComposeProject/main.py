#!/usr/bin/env python3
"""
PINN-Kriging 耦合系统主入口脚本
Main entry script for PINN-Kriging coupling system

用法示例：
1. 使用默认配置：python main.py
2. 使用预设配置：python main.py --preset kriging_only
3. 快速测试：python main.py --preset quick_test
4. 使用随机采样：python main.py --preset random_sampling
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import time
import json

# 将项目根目录和src目录添加到Python路径中
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent)) # PINN_ts 目录
sys.path.insert(0, str(current_dir / 'src')) # ComposeProject/src 目录
sys.path.insert(0, str(current_dir)) # ComposeProject 目录

# 从我们重构的模块中导入
from config import load_config_dict
from src.data.loader import (
    load_3d_data_from_sheets, 
    process_grid_to_dose_data, 
    sample_training_points, 
    sample_kriging_style,
    create_prediction_grid
)
from src.models.pinn import PINNModel
from src.analysis.plotting import plot_training_comparison
from src.utils.display import print_compose_banner
from src.utils.environment import validate_compose_environment


def get_training_samples(dose_data, config):
    """
    根据配置获取训练样本
    
    支持两种采样策略：
    1. kriging_style: Kriging风格的结构化网格采样
    2. 其他策略: 使用 sample_training_points 进行随机采样
    
    Args:
        dose_data: 处理后的剂量数据字典
        config: 配置字典
        
    Returns:
        (train_points, train_values): 训练点坐标和对应的值
    """
    sampling_cfg = config.get('sampling', {})
    strategy = sampling_cfg.get('strategy', 'positive_only')
    
    print(f"\n--- 📊 采样策略: {strategy} ---")
    
    if strategy == 'kriging_style':
        # 使用Kriging风格采样
        kriging_cfg = sampling_cfg.get('kriging_style', {})
        
        print(f"  - 采样区域起点: {kriging_cfg.get('box_origin', [5, 5, 5])}")
        print(f"  - 采样区域延伸: {kriging_cfg.get('box_extent', [90, 90, 90])}")
        print(f"  - 采样步长: {kriging_cfg.get('step_sizes', [5])}")
        
        train_points, train_values = sample_kriging_style(
            dose_data,
            box_origin=kriging_cfg.get('box_origin', [5, 5, 5]),
            box_extent=kriging_cfg.get('box_extent', [90, 90, 90]),
            step_sizes=kriging_cfg.get('step_sizes', [5]),
            source_positions=kriging_cfg.get('source_positions', None),
            source_exclusion_radius=kriging_cfg.get('source_exclusion_radius', 30.0)
        )
    else:
        # 使用随机采样
        random_cfg = sampling_cfg.get('random_sampling', {})
        num_samples = random_cfg.get('num_samples', config.get('data', {}).get('num_samples', 300))
        
        print(f"  - 采样数量: {num_samples}")
        
        train_points, train_values = sample_training_points(
            dose_data=dose_data,
            num_samples=num_samples,
            strategy=strategy if strategy != 'kriging_style' else 'positive_only'
        )
    
    print(f"  ✅ 采样完成: {len(train_points)} 个训练点")
    return train_points, train_values


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模块化的PINN耦合系统")
    parser.add_argument('--preset', type=str, default='default', 
                       help='指定要使用的config.py中的预设配置')
    parser.add_argument('--verbose', action='store_true',
                       help='打印详细的配置信息')
    args = parser.parse_args()

    print_compose_banner()
    
    # 1. 验证依赖
    dep_status = validate_compose_environment()
    print("\n--- 📦 依赖状态检查 ---")
    for dep, status in dep_status.items():
        print(f"  - {dep}: {'✅ 可用' if status else '❌ 不可用'}")
    
    # 2. 加载配置
    print(f"\n--- ⚙️ 正在加载配置 (预设: {args.preset}) ---")
    config = load_config_dict(args.preset)
    np.random.seed(config.get('system', {}).get('random_seed', 42))
    
    # 调试：打印完整的配置字典
    if args.verbose:
        print("--- 调试信息：当前使用的完整配置 ---")
        print(json.dumps(config, indent=2, default=str))
        print("------------------------------------")
    
    # 3. 数据加载和处理
    print("\n--- 💾 正在加载数据 ---")
    data_cfg = config.get('data', {})
    dose_grid = load_3d_data_from_sheets(
        file_path=data_cfg.get('file_path'),
        sheet_name_template=data_cfg.get('sheet_name_template'),
        use_cols=data_cfg.get('use_columns'),
        z_size=data_cfg.get('z_size'),
        y_size=data_cfg.get('y_size'),
    )
    
    dose_data = process_grid_to_dose_data(
        dose_grid=dose_grid,
        space_dims=data_cfg.get('space_dims')
    )
    
    # 4. 使用统一的采样函数
    train_points, train_values = get_training_samples(dose_data, config)
    
    # 5. 创建预测网格
    prediction_points = create_prediction_grid(
        dose_data=dose_data,
        downsample_factor=data_cfg.get('downsample_factor', 1)
    )

    # 6. 初始化并训练PINN模型
    print("\n--- 🚀 正在执行 PINN 工作流 ---")
    start_time = time.time()
    
    pinn_config = config.get('pinn', {})
    
    # 准备test_data
    true_field_values = dose_data['dose_grid'].flatten()
    dummy_test_data = np.hstack([prediction_points, true_field_values[:len(prediction_points)].reshape(-1, 1)])

    pinn_training_data = np.hstack([train_points, train_values])

    model = PINNModel(
        dose_data=dose_data,
        training_data=pinn_training_data,
        test_data=dummy_test_data, 
        **pinn_config.get('model_params', {})
    )
    
    # 从配置中提取训练参数并生成配点
    training_params = pinn_config.get('training_params', {})
    model_params = pinn_config.get('model_params', {})
    num_collocation = model_params.get('num_collocation_points', 4096)
    
    print(f"INFO: Generating {num_collocation} collocation points for training cycle...")
    collocation_points = np.random.uniform(
        low=dose_data['world_min'],
        high=dose_data['world_max'],
        size=(num_collocation, 3)
    )
    
    model.run_training_cycle(
        max_epochs=training_params.get('total_epochs', 5000),
        detect_every=training_params.get('detect_every', 500),
        collocation_points=collocation_points,
        checkpoint_path_prefix=config.get('system', {}).get('checkpoint_path', './models/pinn_checkpoint')
    )
    
    total_time = time.time() - start_time
    print(f"\n--- ✅ 工作流执行完毕 ---")
    print(f"  - 总耗时: {total_time:.2f} 秒")
    print(f"  - 训练点数: {len(train_points)}")

    # 7. 分析和保存
    print("\n--- 📈 正在分析与保存结果 ---")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_name = config.get('experiment', {}).get('name', 'default')

    if hasattr(model, 'mre_history') and model.epoch_history:
        history = {'高级PINN': {'epochs': model.epoch_history, 'metrics': model.mre_history}}
        plot_training_comparison(
            history,
            title=f"PINN训练历史 ({exp_name})",
            save_path=results_dir / f"training_history_{exp_name}.png"
        )
    
    print("\n🎉 所有流程执行完毕。")


if __name__ == "__main__":
    main()
