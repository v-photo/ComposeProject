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
import json

# 将项目根目录和src目录添加到Python路径中
# 这使得我们可以使用绝对导入，如 from src.workflows.auto_selection import ...
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent)) # PINN_ts 目录
sys.path.insert(0, str(current_dir / 'src')) # ComposeProject/src 目录
sys.path.insert(0, str(current_dir)) # ComposeProject 目录

# 从我们重构的模块中导入
from config import load_config_dict
from src.data.loader import load_3d_data_from_sheets, process_grid_to_dose_data, sample_training_points, create_prediction_grid
from src.models.pinn import PINNModel
from src.analysis.plotting import plot_training_comparison
from src.utils.display import print_compose_banner
from src.utils.environment import validate_compose_environment

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模块化的PINN耦合系统")
    parser.add_argument('--preset', type=str, default='default', help='指定要使用的config.py中的预设配置')
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
    print("--- 调试信息：当前使用的完整配置 ---")
    print(json.dumps(config, indent=2))
    print("------------------------------------")
    
    # 3. 数据加载、处理和采样
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
    
    train_points, train_values = sample_training_points(
        dose_data=dose_data,
        num_samples=data_cfg.get('num_samples')
    )
    
    prediction_points = create_prediction_grid(
        dose_data=dose_data,
        downsample_factor=data_cfg.get('downsample_factor')
    )

    # 4. 初始化并训练PINN模型
    print("\n--- 🚀 正在执行 PINN 工作流 ---")
    start_time = time.time()
    
    pinn_config = config.get('pinn', {})
    
    # 准备test_data，这里的test_data是全场的真值网格
    # PINNModel内部会使用它来计算MRE
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
    num_collocation = model_params.get('num_collocation_points')
    
    print(f"INFO: Generating {num_collocation} collocation points for training cycle...")
    collocation_points = np.random.uniform(
        low=dose_data['world_min'],
        high=dose_data['world_max'],
        size=(num_collocation, 3)
    )
    
    model.run_training_cycle(
        max_epochs=training_params.get('total_epochs'),
        detect_every=training_params.get('detect_every'),
        collocation_points=collocation_points,
        checkpoint_path_prefix=config.get('system', {}).get('checkpoint_path')
    )
    
    total_time = time.time() - start_time
    print(f"\n--- ✅ 工作流执行完毕 ---")
    print(f"  - 总耗时: {total_time:.2f} 秒")

    # 5. 分析和保存
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