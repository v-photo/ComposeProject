#!/usr/bin/env python3
"""
GPU Block-Kriging × PINN 耦合重建主程序
GPU-Accelerated Block Kriging × PINN Coupling Main Program

支持三种运行模式:
- common: 通用工具演示和环境检查
- mode1: 方案1演示 (PINN → 残差Kriging → 加权融合)
- mode2: 方案2演示 (Kriging ROI样本扩充 → PINN重训练)

用法:
    python main.py --mode common
    python main.py --mode mode1 --num_samples 300 --fusion_weight 0.6
    python main.py --mode mode2 --roi_strategy high_density --augment_factor 2.5

作者: AI Assistant
日期: 2024
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
from ComposeTools import (
    ComposeConfig, FieldTensor, ProbeSet,
    DataNormalizer, MetricsCalculator, VisualizationTools,
    KrigingAdapter, PINNAdapter, CouplingWorkflow,
    validate_compose_environment, print_compose_banner
)

def generate_synthetic_3d_data(n_samples: int = 300, 
                              space_dims: list = None,
                              noise_level: float = 0.05,
                              random_seed: int = 42) -> tuple:
    """
    生成合成3D辐射场数据用于演示
    Generate synthetic 3D radiation field data for demonstration
    
    Args:
        n_samples: 采样点数量
        space_dims: 空间维度 [x, y, z]
        noise_level: 噪声水平
        random_seed: 随机种子
        
    Returns:
        (train_points, train_values, test_points, test_values, field_info)
    """
    if space_dims is None:
        space_dims = [20.0, 10.0, 10.0]
    
    np.random.seed(random_seed)
    
    # 定义辐射源位置和强度
    source_positions = np.array([
        [2.0, 0.0, 0.0],   # 主源
        [-3.0, 2.0, 1.0],  # 次源
        [1.0, -2.5, -1.5]  # 弱源
    ])
    source_strengths = np.array([100.0, 50.0, 25.0])
    
    # 世界坐标边界
    world_min = np.array([-10.0, -5.0, -5.0])
    world_max = np.array([10.0, 5.0, 5.0])
    
    # 生成训练采样点（随机采样）
    train_points = np.random.rand(n_samples, 3)
    train_points = world_min + train_points * (world_max - world_min)
    
    # 生成测试网格点（规则网格）
    test_grid_shape = (20, 15, 15)
    x_test = np.linspace(world_min[0], world_max[0], test_grid_shape[0])
    y_test = np.linspace(world_min[1], world_max[1], test_grid_shape[1])
    z_test = np.linspace(world_min[2], world_max[2], test_grid_shape[2])
    
    X, Y, Z = np.meshgrid(x_test, y_test, z_test, indexing='ij')
    test_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    def compute_radiation_field(points, sources, strengths):
        """计算多源辐射场 (简化的反平方律模型)"""
        field_values = np.zeros(len(points))
        
        for source_pos, strength in zip(sources, strengths):
            # 计算距离
            distances = np.linalg.norm(points - source_pos, axis=1)
            # 防止除零，添加小的衰减常数
            distances = np.maximum(distances, 0.1)
            # 反平方律 + 指数衰减
            field_contribution = strength / (distances**2) * np.exp(-distances / 5.0)
            field_values += field_contribution
        
        # 添加背景噪声
        field_values += np.random.normal(0, noise_level * np.mean(field_values), len(points))
        field_values = np.maximum(field_values, 1e-6)  # 确保非负
        
        return field_values
    
    # 计算场值
    train_values = compute_radiation_field(train_points, source_positions, source_strengths)
    test_values = compute_radiation_field(test_points, source_positions, source_strengths)
    
    field_info = {
        'space_dims': space_dims,
        'world_bounds': {'min': world_min, 'max': world_max},
        'source_positions': source_positions,
        'source_strengths': source_strengths,
        'test_grid_shape': test_grid_shape,
        'noise_level': noise_level
    }
    
    print(f"✅ 生成合成数据完成:")
    print(f"   - 训练样本: {len(train_points)} 个点")
    print(f"   - 测试网格: {len(test_points)} 个点 ({test_grid_shape})")
    print(f"   - 训练值范围: [{np.min(train_values):.2e}, {np.max(train_values):.2e}]")
    print(f"   - 测试值范围: [{np.min(test_values):.2e}, {np.max(test_values):.2e}]")
    
    return train_points, train_values, test_points, test_values, field_info

def load_real_data_from_excel(data_file_path: str = "../PINN/DATA.xlsx") -> tuple:
    """
    加载真实的辐射场数据从PINN/DATA.xlsx
    Load real radiation field data from PINN/DATA.xlsx
    
    Args:
        data_file_path: DATA.xlsx文件路径
        
    Returns:
        (train_points, train_values, test_points, test_values, field_info)
    """
    import sys
    from pathlib import Path
    
    # 添加PINN目录到路径以导入dataAnalysis
    pinn_dir = Path(__file__).parent.parent / "PINN"
    sys.path.insert(0, str(pinn_dir))
    
    try:
        from dataAnalysis import get_data
        print("✅ 成功导入dataAnalysis模块")
    except ImportError as e:
        print(f"❌ 无法导入dataAnalysis模块: {e}")
        raise
    
    # 加载真实数据
    print(f"🔄 正在加载真实数据: {data_file_path}")
    data_file_full_path = str(pinn_dir / "DATA.xlsx")
    
    try:
        data_dict = get_data(data_file_full_path)
        print(f"✅ 成功加载数据，包含 {len(data_dict)} 个z层")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise
    
    # 使用PINN的RadiationDataProcessor处理数据
    try:
        from tools import RadiationDataProcessor, DataLoader
        print("✅ 成功导入PINN tools模块")
    except ImportError as e:
        print(f"❌ 无法导入PINN tools模块: {e}")
        raise
    
    # 处理数据
    dose_data = DataLoader.load_dose_from_dict(
        data_dict=data_dict,
        space_dims=[20.0, 10.0, 10.0]  # 根据实际物理尺寸调整
    )
    
    # 采样训练数据（使用positive_only策略，避免零值）
    train_points, train_values, _ = DataLoader.sample_training_points(
        dose_data, 
        num_samples=300, 
        sampling_strategy='positive_only'
    )
    
    # 采样测试数据（更少的点用于测试）
    test_points, test_values, _ = DataLoader.sample_training_points(
        dose_data,
        num_samples=150,
        sampling_strategy='positive_only'
    )
    
    # 字段信息
    field_info = {
        'space_dims': dose_data['space_dims'].tolist(),
        'world_bounds': {
            'min': dose_data['world_min'],
            'max': dose_data['world_max']
        },
        'grid_shape': dose_data['grid_shape'],
        'dose_data': dose_data,  # 保存完整的dose_data供后续使用
        'data_source': 'real_excel_data'
    }
    
    print(f"✅ 数据处理完成:")
    print(f"   - 训练样本: {len(train_points)} 个点")
    print(f"   - 测试样本: {len(test_points)} 个点")
    print(f"   - 训练值范围: [{np.min(train_values):.2e}, {np.max(train_values):.2e}]")
    print(f"   - 测试值范围: [{np.min(test_values):.2e}, {np.max(test_values):.2e}]")
    print(f"   - 网格形状: {dose_data['grid_shape']}")
    print(f"   - 空间尺寸: {dose_data['space_dims']}")
    
    return train_points, train_values, test_points, test_values, field_info

def run_common_mode(args):
    """
    运行通用模式: 环境检查和工具演示
    Run common mode: environment check and tools demonstration
    """
    print("\n" + "="*60)
    print("🔧 通用模式: 环境检查和工具演示")
    print("="*60)
    
    # 环境检查
    print("\n1️⃣ 环境完整性检查...")
    env_status = validate_compose_environment()
    
    # 加载数据 - 默认使用真实数据
    print("\n2️⃣ 加载数据...")
    try:
        print("使用真实DATA.xlsx数据")
        train_points, train_values, test_points, test_values, field_info = load_real_data_from_excel(
            data_file_path=args.data_file
        )
    except Exception as e:
        print(f"⚠️ 真实数据加载失败: {e}")
        print("回退到合成演示数据")
        train_points, train_values, test_points, test_values, field_info = generate_synthetic_3d_data(
            n_samples=args.num_samples,
            noise_level=args.noise_level,
            random_seed=args.random_seed
        )
    
    # 数据结构演示
    print("\n3️⃣ 数据结构标准化演示...")
    field_tensor = FieldTensor(
        coordinates=train_points,
        values=train_values,
        metadata={'type': 'radiation_field', 'units': 'arbitrary'}
    )
    
    probe_set = ProbeSet(
        positions=train_points,
        measurements=train_values,
        metadata={'sensor_type': 'synthetic', 'calibration': 'simulated'}
    )
    
    print(f"   ✅ FieldTensor: {field_tensor.coordinates.shape} 坐标, {field_tensor.values.shape} 数值")
    print(f"   ✅ ProbeSet: {probe_set.positions.shape} 位置, {probe_set.measurements.shape} 测量值")
    
    # 数据归一化演示
    print("\n4️⃣ 数据归一化演示...")
    normalized_values, norm_info = DataNormalizer.robust_normalize(train_values)
    print(f"   原始值范围: [{np.min(train_values):.2e}, {np.max(train_values):.2e}]")
    print(f"   归一化后范围: [{np.min(normalized_values):.3f}, {np.max(normalized_values):.3f}]")
    print(f"   归一化参数: {norm_info}")
    
    # 误差统计演示
    print("\n5️⃣ 误差统计演示...")
    # 创建模拟预测值（添加一些误差）
    pred_values = train_values * (1 + np.random.normal(0, 0.1, len(train_values)))
    metrics = MetricsCalculator.compute_metrics(train_values, pred_values)
    print("   预测误差指标:")
    for metric, value in metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # 可视化演示
    print("\n6️⃣ 可视化功能演示...")
    
    # 转换为3D网格用于可视化
    test_grid_shape = field_info['test_grid_shape']
    true_grid = test_values.reshape(test_grid_shape)
    pred_grid = test_values.reshape(test_grid_shape) * (1 + np.random.normal(0, 0.05, test_grid_shape))
    
    # 绘制对比图
    fig = VisualizationTools.plot_comparison_2d_slice(
        true_grid, pred_grid, slice_axis=2, slice_idx=test_grid_shape[2]//2,
        title_prefix="演示数据 - "
    )
    
    if args.save_plots:
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        fig.savefig(plots_dir / "common_mode_comparison.png", dpi=300, bbox_inches='tight')
        print(f"   📊 对比图已保存: {plots_dir / 'common_mode_comparison.png'}")
    
    plt.show()
    
    # 残差分析演示
    residuals = pred_values - train_values
    fig_residual = VisualizationTools.plot_residual_analysis(residuals, train_points)
    
    if args.save_plots:
        fig_residual.savefig(plots_dir / "common_mode_residuals.png", dpi=300, bbox_inches='tight')
        print(f"   📊 残差分析图已保存: {plots_dir / 'common_mode_residuals.png'}")
    
    plt.show()
    
    print("\n✅ 通用模式演示完成!")
    return env_status

def run_mode1(args):
    """
    运行方案1: PINN → 残差Kriging → 加权融合
    Run Mode 1: PINN → Residual Kriging → Weighted Fusion
    """
    print("\n" + "="*60)
    print("🚀 方案1: PINN → 残差Kriging → 加权融合")
    print("="*60)
    
    # 加载数据
    print("\n📊 加载数据...")
    if args.use_real_data:
        print("使用真实DATA.xlsx数据")
        train_points, train_values, test_points, test_values, field_info = load_real_data_from_excel(
            data_file_path=args.data_file
        )
    else:
        print("生成合成演示数据")
        train_points, train_values, test_points, test_values, field_info = generate_synthetic_3d_data(
            n_samples=args.num_samples,
            noise_level=args.noise_level,
            random_seed=args.random_seed
        )
    
    # 配置耦合系统
    config = ComposeConfig(
        gpu_enabled=args.gpu_enabled,
        verbose=args.verbose,
        random_seed=args.random_seed,
        fusion_weight=args.fusion_weight,
        pinn_epochs=args.pinn_epochs,
        kriging_variogram_model=args.variogram_model
    )
    
    # 创建工作流
    workflow = CouplingWorkflow(config)
    
    # 执行方案1流程
    print(f"\n🔄 执行方案1流程 (融合权重={args.fusion_weight})...")
    
    start_time = time.time()
    
    try:
        results = workflow.run_mode1_pipeline(
            train_points=train_points,
            train_values=train_values,
            prediction_points=test_points,
            fusion_weight=args.fusion_weight,
            space_dims=field_info['space_dims'],
            world_bounds=field_info['world_bounds'],
            kriging_params={'variogram_model': args.variogram_model},
            epochs=args.pinn_epochs,
            max_training_points=1000,  # 限制最大训练点数避免内存问题
            network_config={'layers': [3, 32, 32, 32, 1]}  # 使用安全的网络配置
        )
        
        execution_time = time.time() - start_time
        print(f"\n⏱️ 方案1执行时间: {execution_time:.2f} 秒")
        
        # 评估结果
        print("\n📈 结果评估...")
        final_predictions = results['final_predictions']
        
        # 计算各种预测的误差指标
        pinn_metrics = MetricsCalculator.compute_metrics(test_values, results['pinn_predictions'])
        final_metrics = MetricsCalculator.compute_metrics(test_values, final_predictions)
        
        print("\n📊 PINN基线性能:")
        for metric, value in pinn_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        print("\n📊 方案1融合后性能:")
        for metric, value in final_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        print("\n📊 性能提升:")
        for metric in pinn_metrics:
            if metric in ['MAE', 'RMSE', 'MAPE']:  # 越小越好
                improvement = (pinn_metrics[metric] - final_metrics[metric]) / pinn_metrics[metric] * 100
                print(f"   {metric} 改善: {improvement:+.2f}%")
            elif metric == 'R2':  # 越大越好
                improvement = (final_metrics[metric] - pinn_metrics[metric]) / abs(pinn_metrics[metric]) * 100
                print(f"   {metric} 改善: {improvement:+.2f}%")
        
        # 可视化结果
        if args.save_plots or args.show_plots:
            print("\n🎨 生成可视化结果...")
            
            test_grid_shape = field_info['test_grid_shape']
            true_grid = test_values.reshape(test_grid_shape)
            pinn_grid = results['pinn_predictions'].reshape(test_grid_shape)
            final_grid = final_predictions.reshape(test_grid_shape)
            
            # 对比图1: 真实 vs PINN
            fig1 = VisualizationTools.plot_comparison_2d_slice(
                true_grid, pinn_grid, slice_axis=2, slice_idx=test_grid_shape[2]//2,
                title_prefix="方案1 - PINN基线 - "
            )
            
            # 对比图2: 真实 vs 融合结果
            fig2 = VisualizationTools.plot_comparison_2d_slice(
                true_grid, final_grid, slice_axis=2, slice_idx=test_grid_shape[2]//2,
                title_prefix="方案1 - 融合结果 - "
            )
            
            # 残差分析
            final_residuals = final_predictions - test_values
            fig3 = VisualizationTools.plot_residual_analysis(final_residuals, test_points)
            
            if args.save_plots:
                plots_dir = Path("plots")
                plots_dir.mkdir(exist_ok=True)
                fig1.savefig(plots_dir / "mode1_pinn_baseline.png", dpi=300, bbox_inches='tight')
                fig2.savefig(plots_dir / "mode1_fusion_result.png", dpi=300, bbox_inches='tight')
                fig3.savefig(plots_dir / "mode1_residual_analysis.png", dpi=300, bbox_inches='tight')
                print(f"   📊 可视化结果已保存至 {plots_dir}/")
            
            if args.show_plots:
                plt.show()
        
        print("\n✅ 方案1演示完成!")
        return results
        
    except Exception as e:
        print(f"\n❌ 方案1执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

def run_mode2(args):
    """
    运行方案2: Kriging ROI样本扩充 → PINN重训练
    Run Mode 2: Kriging ROI Sample Augmentation → PINN Retraining
    """
    print("\n" + "="*60)
    print("🎯 方案2: Kriging ROI样本扩充 → PINN重训练")
    print("="*60)
    
    # 加载数据
    print("\n📊 加载数据...")
    if args.use_real_data:
        print("使用真实DATA.xlsx数据")
        train_points, train_values, test_points, test_values, field_info = load_real_data_from_excel(
            data_file_path=args.data_file
        )
    else:
        print("生成合成演示数据")
        train_points, train_values, test_points, test_values, field_info = generate_synthetic_3d_data(
            n_samples=args.num_samples,
            noise_level=args.noise_level,
            random_seed=args.random_seed
        )
    
    # 配置耦合系统
    config = ComposeConfig(
        gpu_enabled=args.gpu_enabled,
        verbose=args.verbose,
        random_seed=args.random_seed,
        roi_detection_strategy=args.roi_strategy,
        sample_augment_factor=args.augment_factor,
        pinn_epochs=args.pinn_epochs,
        kriging_variogram_model=args.variogram_model
    )
    
    # 创建工作流
    workflow = CouplingWorkflow(config)
    
    # 首先训练PINN基线用于对比
    print("\n🔥 训练PINN基线模型...")
    baseline_pinn = PINNAdapter(config)
    baseline_pinn.fit(train_points, train_values, 
                     space_dims=field_info['space_dims'],
                     world_bounds=field_info['world_bounds'])
    baseline_predictions = baseline_pinn.predict(test_points)
    
    # 执行方案2流程
    print(f"\n🔄 执行方案2流程 (ROI策略={args.roi_strategy}, 扩充倍数={args.augment_factor})...")
    
    start_time = time.time()
    
    try:
        results = workflow.run_mode2_pipeline(
            train_points=train_points,
            train_values=train_values,
            prediction_points=test_points,
            roi_strategy=args.roi_strategy,
            augment_factor=args.augment_factor,
            space_dims=field_info['space_dims'],
            world_bounds=field_info['world_bounds'],
            roi_params={'density_percentile': 70, 'expansion_factor': 1.3},
            kriging_params={'variogram_model': args.variogram_model},
            epochs=args.pinn_epochs
        )
        
        execution_time = time.time() - start_time
        print(f"\n⏱️ 方案2执行时间: {execution_time:.2f} 秒")
        
        # 评估结果
        print("\n📈 结果评估...")
        final_predictions = results['final_predictions']
        
        # 计算各种预测的误差指标
        baseline_metrics = MetricsCalculator.compute_metrics(test_values, baseline_predictions)
        final_metrics = MetricsCalculator.compute_metrics(test_values, final_predictions)
        
        print("\n📊 PINN基线性能:")
        for metric, value in baseline_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        print("\n📊 方案2增强后性能:")
        for metric, value in final_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        print("\n📊 性能提升:")
        for metric in baseline_metrics:
            if metric in ['MAE', 'RMSE', 'MAPE']:  # 越小越好
                improvement = (baseline_metrics[metric] - final_metrics[metric]) / baseline_metrics[metric] * 100
                print(f"   {metric} 改善: {improvement:+.2f}%")
            elif metric == 'R2':  # 越大越好
                improvement = (final_metrics[metric] - baseline_metrics[metric]) / abs(baseline_metrics[metric]) * 100
                print(f"   {metric} 改善: {improvement:+.2f}%")
        
        # 样本扩充统计
        original_count = len(train_points)
        augmented_count = len(results['augmented_points'])
        print(f"\n📈 样本扩充统计:")
        print(f"   原始样本数: {original_count}")
        print(f"   扩充后样本数: {augmented_count}")
        print(f"   扩充倍数: {augmented_count / original_count:.2f}")
        
        # ROI信息
        roi_bounds = results['roi_bounds']
        print(f"\n🎯 ROI检测结果:")
        print(f"   策略: {args.roi_strategy}")
        print(f"   ROI边界: {roi_bounds['min']} 到 {roi_bounds['max']}")
        if 'mask' in roi_bounds:
            roi_point_count = np.sum(roi_bounds['mask'])
            print(f"   ROI内训练点数: {roi_point_count}/{original_count}")
        
        # 可视化结果
        if args.save_plots or args.show_plots:
            print("\n🎨 生成可视化结果...")
            
            test_grid_shape = field_info['test_grid_shape']
            true_grid = test_values.reshape(test_grid_shape)
            baseline_grid = baseline_predictions.reshape(test_grid_shape)
            final_grid = final_predictions.reshape(test_grid_shape)
            
            # 对比图1: 真实 vs PINN基线
            fig1 = VisualizationTools.plot_comparison_2d_slice(
                true_grid, baseline_grid, slice_axis=2, slice_idx=test_grid_shape[2]//2,
                title_prefix="方案2 - PINN基线 - "
            )
            
            # 对比图2: 真实 vs 增强结果
            fig2 = VisualizationTools.plot_comparison_2d_slice(
                true_grid, final_grid, slice_axis=2, slice_idx=test_grid_shape[2]//2,
                title_prefix="方案2 - 增强结果 - "
            )
            
            # 残差分析
            final_residuals = final_predictions - test_values
            fig3 = VisualizationTools.plot_residual_analysis(final_residuals, test_points)
            
            if args.save_plots:
                plots_dir = Path("plots")
                plots_dir.mkdir(exist_ok=True)
                fig1.savefig(plots_dir / "mode2_pinn_baseline.png", dpi=300, bbox_inches='tight')
                fig2.savefig(plots_dir / "mode2_enhanced_result.png", dpi=300, bbox_inches='tight')
                fig3.savefig(plots_dir / "mode2_residual_analysis.png", dpi=300, bbox_inches='tight')
                print(f"   📊 可视化结果已保存至 {plots_dir}/")
            
            if args.show_plots:
                plt.show()
        
        print("\n✅ 方案2演示完成!")
        return results
        
    except Exception as e:
        print(f"\n❌ 方案2执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="GPU Block-Kriging × PINN 耦合重建演示程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --mode common                          # 通用工具演示
  python main.py --mode mode1 --fusion_weight 0.7      # 方案1演示
  python main.py --mode mode2 --roi_strategy high_value # 方案2演示
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--mode', 
        choices=['common', 'mode1', 'mode2'],
        required=True,
        help='运行模式: common(通用工具) | mode1(残差融合) | mode2(样本扩充)'
    )
    
    # 数据源选择
    parser.add_argument('--use_real_data', action='store_true', default=True, 
                       help='使用真实DATA.xlsx数据而非合成数据')
    parser.add_argument('--use_synthetic_data', action='store_true', default=False,
                       help='使用合成数据而非真实数据')
    parser.add_argument('--data_file', type=str, default="../PINN/DATA.xlsx",
                       help='数据文件路径 (默认: ../PINN/DATA.xlsx)')
    
    # 通用参数
    parser.add_argument('--num_samples', type=int, default=300, help='训练样本数量 (默认: 300)')
    parser.add_argument('--noise_level', type=float, default=0.05, help='数据噪声水平 (默认: 0.05)')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--gpu_enabled', action='store_true', default=True, help='启用GPU加速')
    parser.add_argument('--no_gpu', dest='gpu_enabled', action='store_false', help='禁用GPU加速')
    parser.add_argument('--verbose', action='store_true', default=True, help='详细输出')
    parser.add_argument('--quiet', dest='verbose', action='store_false', help='简洁输出')
    
    # PINN参数
    parser.add_argument('--pinn_epochs', type=int, default=500, help='PINN训练轮数 (默认: 500)')
    
    # Kriging参数
    parser.add_argument('--variogram_model', choices=['linear', 'exponential', 'gaussian'], 
                       default='linear', help='变异函数模型 (默认: linear)')
    
    # 方案1专用参数
    parser.add_argument('--fusion_weight', type=float, default=0.5, 
                       help='方案1融合权重 ω ∈ (0,1) (默认: 0.5)')
    
    # 方案2专用参数
    parser.add_argument('--roi_strategy', choices=['high_density', 'high_value', 'bounding_box'],
                       default='high_density', help='方案2 ROI检测策略 (默认: high_density)')
    parser.add_argument('--augment_factor', type=float, default=2.0,
                       help='方案2样本扩充倍数 (默认: 2.0)')
    
    # 可视化参数
    parser.add_argument('--save_plots', action='store_true', help='保存可视化图片')
    parser.add_argument('--show_plots', action='store_true', help='显示可视化图片')
    parser.add_argument('--no_plots', dest='show_plots', action='store_false', help='不显示图片')
    
    return parser

def main():
    """主函数"""
    print_compose_banner()
    
    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 显示配置信息
    print("\n" + "="*60)
    print("🛠️  运行配置")
    print("="*60)
    print(f"运行模式: {args.mode}")
    print(f"样本数量: {args.num_samples}")
    print(f"GPU加速: {'启用' if args.gpu_enabled else '禁用'}")
    print(f"详细输出: {'是' if args.verbose else '否'}")
    print(f"随机种子: {args.random_seed}")
    
    if args.mode in ['mode1', 'mode2']:
        print(f"PINN训练轮数: {args.pinn_epochs}")
        print(f"变异函数模型: {args.variogram_model}")
    
    if args.mode == 'mode1':
        print(f"融合权重: {args.fusion_weight}")
    elif args.mode == 'mode2':
        print(f"ROI策略: {args.roi_strategy}")
        print(f"扩充倍数: {args.augment_factor}")
    
    # 执行相应模式
    try:
        if args.mode == 'common':
            results = run_common_mode(args)
        elif args.mode == 'mode1':
            results = run_mode1(args)
        elif args.mode == 'mode2':
            results = run_mode2(args)
        else:
            print(f"❌ 不支持的运行模式: {args.mode}")
            return 1
        
        if results is not None:
            print(f"\n🎉 {args.mode} 模式运行成功!")
        else:
            print(f"\n⚠️ {args.mode} 模式运行结束，部分功能可能未完全工作")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⛔ 用户中断程序")
        return 130
    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 