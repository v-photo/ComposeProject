import sys
print("--- SCRIPT START ---")
import os
import argparse
import numpy as np
from pathlib import Path
# import matplotlib
# matplotlib.use('Agg') # <--- 在导入pyplot之前设置后端
# import matplotlib.pyplot as plt

# --- 路径设置 ---
try:
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent
except NameError:
    project_root = Path('.').resolve()

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'PINN'))
sys.path.insert(0, str(project_root / 'ComposeProject'))

# --- 模块导入 ---
try:
    from ComposeTools import (
        ComposeConfig,
        CouplingWorkflow,
        print_compose_banner,
        validate_compose_environment,
        MetricsCalculator,
        VisualizationTools
    )
    from PINN.data_processing import DataLoader
    from PINN.dataAnalysis import get_data
    print("✅ 模块导入成功。")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def main(args):
    """主执行函数"""
    
    # ==================== 1. 初始化和环境检查 ====================
    print_compose_banner()
    if not all(validate_compose_environment().values()):
        print("❌ 环境检查失败，请检查依赖项。程序退出。")
        return

    config = ComposeConfig(random_seed=args.seed)
    
    # PINN 物理参数 (应从更可靠的来源获取，此处为示例)
    physical_params = {
        'rho_material': 1.205,
        'mass_energy_abs_coeff': 1.0
    }

    # ==================== 2. 数据加载和准备 ====================
    print(f"\n" + "="*25 + " 步骤1: 数据加载 " + "="*25)
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        return
        
    raw_data = get_data(args.data_path)
    dose_data = DataLoader.load_dose_from_dict(
        data_dict=raw_data,
        space_dims=np.array([20.0, 10.0, 10.0]) # 示例维度
    )
    
    # 采样训练点 (用于两个方案的初始训练)
    print(f"采样 {args.num_samples} 个训练点...")
    train_points, train_values, _ = DataLoader.sample_training_points(
        dose_data, 
        num_samples=args.num_samples,
        sampling_strategy='positive_only' # 使用 'positive_only' 策略
    )
    print(f"✅ 成功采样 {len(train_points)} 个训练点。")

    # 准备全场预测点
    original_grid_shape = np.array(dose_data['dose_grid'].shape)
    if args.downsample > 1:
        print(f"⚠️ 警告: 预测网格将通过系数 {args.downsample} 进行降采样以加速调试。")
        step = int(args.downsample)
        pred_x_indices = np.arange(0, original_grid_shape[0], step)
        pred_y_indices = np.arange(0, original_grid_shape[1], step)
        pred_z_indices = np.arange(0, original_grid_shape[2], step)
        grid_shape = (len(pred_x_indices), len(pred_y_indices), len(pred_z_indices))
    else:
        pred_x_indices = np.arange(original_grid_shape[0])
        pred_y_indices = np.arange(original_grid_shape[1])
        pred_z_indices = np.arange(original_grid_shape[2])
        grid_shape = original_grid_shape

    pred_x = dose_data['world_min'][0] + (pred_x_indices + 0.5) * dose_data['voxel_size'][0]
    pred_y = dose_data['world_min'][1] + (pred_y_indices + 0.5) * dose_data['voxel_size'][1]
    pred_z = dose_data['world_min'][2] + (pred_z_indices + 0.5) * dose_data['voxel_size'][2]
    XX, YY, ZZ = np.meshgrid(pred_x, pred_y, pred_z, indexing='ij')
    prediction_points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T

    # ==================== 3. 初始化并执行工作流 ====================
    print(f"\n" + "="*25 + f" 步骤2: 执行方案 {args.mode} " + "="*25)
    workflow = CouplingWorkflow(physical_params=physical_params, config=config)
    
    results = {}
    pinn_params = {
        'epochs': args.pinn_epochs,
        'use_lbfgs': args.use_lbfgs,
        'loss_weights': [1, 100]
    }

    if args.mode == 1:
        results = workflow.run_mode1_pipeline(
            train_points=train_points,
            train_values=train_values,
            prediction_points=prediction_points,
            dose_data=dose_data,
            **pinn_params
        )
    elif args.mode == 2:
        results = workflow.run_mode2_pipeline(
            train_points=train_points,
            train_values=train_values,
            prediction_points=prediction_points,
            dose_data=dose_data,
            roi_strategy=args.roi_strategy,
            augment_factor=args.augment_factor,
            **pinn_params
        )
    else:
        print(f"❌ 未知的模式: {args.mode}")
        return
        
    print(f"✅ 方案 {args.mode} 执行完毕。")
    
    # ==================== 4. 评估和可视化 ====================
    print(f"\n" + "="*25 + " 步骤3: 结果评估 " + "="*25)
    
    # 准备用于评估的真值
    if args.downsample > 1:
        true_field_for_eval = dose_data['dose_grid'][np.ix_(pred_x_indices, pred_y_indices, pred_z_indices)]
        test_values = true_field_for_eval.flatten()
    else:
        test_values = dose_data['dose_grid'].flatten()

    # ==================== DEBUG: PINN基线 vs 融合结果性能对比 ====================
    print("\n" + "#"*20 + " DEBUG: 性能对比测试 " + "#"*20)
    
    pinn_predictions = results.get('pinn_predictions')
    final_predictions = results.get('final_predictions')

    if pinn_predictions is not None and final_predictions is not None:
        print(f"评估点数: {len(test_values)}")

        # 1. 计算PINN基线性能
        pinn_metrics = MetricsCalculator.compute_metrics(test_values, pinn_predictions)
        print("\n--- PINN基线性能 (无残差修正) ---")
        for name, value in pinn_metrics.items():
            print(f"  - {name}: {value:.6f}")

        # 2. 计算最终融合后性能
        final_metrics = MetricsCalculator.compute_metrics(test_values, final_predictions)
        
        # 根据模式确定标题
        if args.mode == 1:
            print("\n--- 融合后性能 (PINN + Kriging残差修正) ---")
        elif args.mode == 2:
            print("\n--- 增强后性能 (PINN重训练后) ---")
        else:
            print("\n--- 最终性能 ---")
            
        for name, value in final_metrics.items():
            print(f"  - {name}: {value:.6f}")

        # 3. 计算性能提升
        print("\n--- 性能提升分析 ---")
        for metric in pinn_metrics:
            if metric in final_metrics:
                pinn_val = pinn_metrics[metric]
                final_val = final_metrics[metric]
                
                # 对于越小越好的指标 (MAE, RMSE, MAPE)
                if 'MAE' in metric or 'RMSE' in metric or 'MAPE' in metric:
                    if abs(pinn_val) > 1e-9:
                        improvement = (pinn_val - final_val) / pinn_val * 100
                        print(f"  - {metric} 提升: {improvement:+.2f}% (越低越好)")
                # 对于越大越好的指标 (R2)
                elif 'R2' in metric:
                    if abs(pinn_val) > 1e-9:
                        improvement = (final_val - pinn_val) / abs(pinn_val) * 100
                        print(f"  - {metric} 提升: {improvement:+.2f}% (越高越好)")
    else:
        print("⚠️ 未能获取PINN或最终预测结果，无法进行性能对比。")

    print("#"*20 + " DEBUG: 性能对比结束 " + "#"*20 + "\n")
    # =======================================================================
        
    print("\n🎉 所有流程执行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 Kriging-PINN 耦合工作流")
    parser.add_argument('--mode', type=int, choices=[1, 2], default=1,
                        help="选择要运行的耦合方案 (1或2)")
    parser.add_argument('--data_path', type=str, default="PINN/DATA.xlsx",
                        help="输入数据文件的路径 (Excel格式)")
    parser.add_argument('--num_samples', type=int, default=300,
                        help="用于初始训练的采样点数量")
    parser.add_argument('--pinn_epochs', type=int, default=5000,
                        help="PINN训练的周期数")
    parser.add_argument('--downsample', type=int, default=1,
                        help="全场预测网格的降采样系数(>1)，用于加速调试。")
    parser.add_argument('--seed', type=int, default=42,
                        help="随机种子，以确保结果可复现")
    # 为模式2添加新的命令行参数
    parser.add_argument('--roi_strategy', type=str, default='high_density',
                        choices=['high_density', 'high_value', 'bounding_box'],
                        help="[模式2专用] ROI检测策略")
    parser.add_argument('--augment_factor', type=float, default=2.0,
                        help="[模式2专用] Kriging数据增强的样本扩充倍数")
    parser.add_argument('--use_lbfgs', action='store_true',
                        help="在PINN训练中使用L-BFGS进行精细调优")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ 程序执行时发生严重错误: {e}")
        import traceback
        traceback.print_exc()