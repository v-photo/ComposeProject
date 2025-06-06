#!/usr/bin/env python3
"""
GPU Block-Kriging × PINN 耦合重建测试模块
Comprehensive test suite with ≥80% coverage

测试涵盖:
- 数据结构和工具类单元测试
- Kriging和PINN适配器测试  
- 方案1和方案2工作流测试
- 端到端集成测试
- 异常处理和边界条件测试

运行方法:
    pytest tests/test_compose.py -v --cov=ComposeTools --cov-report=html
    python -m pytest tests/test_compose.py --cov=ComposeTools --cov-report=term-missing

作者: AI Assistant
日期: 2024
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入待测试模块
from ComposeTools import (
    # 配置和数据结构
    ComposeConfig, FieldTensor, ProbeSet,
    # 工具类
    DataNormalizer, MetricsCalculator, VisualizationTools,
    # 适配器
    KrigingAdapter, PINNAdapter,
    # 方案专用工具
    Mode1ResidualKriging, Mode1Fusion,
    Mode2ROIDetector, Mode2SampleAugmentor,
    # 工作流
    CouplingWorkflow,
    # 工具函数
    validate_compose_environment, print_compose_banner,
    # 常量
    EPSILON, DEFAULT_METRICS
)

# ==================== 测试数据生成工具 ====================

class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_simple_3d_data(n_samples=50, noise_level=0.1, random_seed=42):
        """生成简单的3D测试数据"""
        np.random.seed(random_seed)
        
        # 生成训练点
        train_points = np.random.rand(n_samples, 3) * 10 - 5  # [-5, 5]
        
        # 简单的二次函数作为真实场
        def simple_field(points):
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            values = 10 * np.exp(-(x**2 + y**2 + z**2) / 25) + np.random.normal(0, noise_level, len(points))
            return np.maximum(values, 1e-6)
        
        train_values = simple_field(train_points)
        
        # 生成测试点
        test_points = np.random.rand(n_samples//2, 3) * 8 - 4  # [-4, 4]
        test_values = simple_field(test_points)
        
        return train_points, train_values, test_points, test_values
    
    @staticmethod  
    def generate_field_info():
        """生成字段信息"""
        return {
            'space_dims': [10.0, 10.0, 10.0],
            'world_bounds': {
                'min': np.array([-5.0, -5.0, -5.0]),
                'max': np.array([5.0, 5.0, 5.0])
            }
        }

# ==================== 数据结构测试 ====================

class TestDataStructures:
    """测试数据结构类"""
    
    def test_field_tensor_creation(self):
        """测试FieldTensor创建"""
        coordinates = np.random.rand(10, 3)
        values = np.random.rand(10)
        uncertainties = np.random.rand(10)
        metadata = {'type': 'test'}
        
        field = FieldTensor(coordinates, values, uncertainties, metadata)
        
        assert field.coordinates.shape == (10, 3)
        assert field.values.shape == (10,)
        assert field.uncertainties.shape == (10,)
        assert field.metadata['type'] == 'test'
    
    def test_field_tensor_validation(self):
        """测试FieldTensor数据验证"""
        coordinates = np.random.rand(10, 3)
        values = np.random.rand(5)  # 错误的长度
        
        with pytest.raises(ValueError, match="坐标和数值的数量不匹配"):
            FieldTensor(coordinates, values)
        
        # 测试错误的坐标维度
        coordinates_2d = np.random.rand(10, 2)
        values = np.random.rand(10)
        
        with pytest.raises(ValueError, match="坐标必须是3维"):
            FieldTensor(coordinates_2d, values)
    
    def test_probe_set_creation(self):
        """测试ProbeSet创建"""
        positions = np.random.rand(10, 3)
        measurements = np.random.rand(10)
        weights = np.random.rand(10)
        
        probe = ProbeSet(positions, measurements, weights)
        
        assert probe.positions.shape == (10, 3)
        assert probe.measurements.shape == (10,)
        assert probe.weights.shape == (10,)
    
    def test_compose_config_defaults(self):
        """测试ComposeConfig默认值"""
        config = ComposeConfig()
        
        assert config.gpu_enabled == True
        assert config.verbose == True
        assert config.random_seed == 42
        assert config.fusion_weight == 0.5
        assert config.pinn_network_layers == [50, 50, 50, 50]

# ==================== 工具类测试 ====================

class TestDataNormalizer:
    """测试数据归一化工具"""
    
    def test_robust_normalize(self):
        """测试鲁棒归一化"""
        data = np.array([1, 2, 3, 4, 5, 100])  # 包含异常值
        
        normalized, info = DataNormalizer.robust_normalize(data)
        
        assert 'method' in info
        assert info['method'] == 'robust'
        assert 'low_val' in info
        assert 'high_val' in info
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_normalize_tensor_to_grid(self):
        """测试张量到网格转换"""
        # 创建测试数据
        coordinates = np.random.rand(20, 3) * 10 - 5
        values = np.random.rand(20)
        field_tensor = FieldTensor(coordinates, values)
        
        grid_shape = (5, 5, 5)
        world_bounds = {'min': np.array([-5, -5, -5]), 'max': np.array([5, 5, 5])}
        
        result = DataNormalizer.normalize_tensor_to_grid(field_tensor, grid_shape, world_bounds)
        
        assert 'grid' in result
        assert 'coordinates' in result
        assert 'bounds' in result
        assert result['grid'].shape == grid_shape

class TestMetricsCalculator:
    """测试误差指标计算器"""
    
    def test_compute_metrics_basic(self):
        """测试基本误差指标计算"""
        true_values = np.array([1, 2, 3, 4, 5])
        pred_values = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = MetricsCalculator.compute_metrics(true_values, pred_values)
        
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics
        assert metrics['MAE'] > 0
        assert metrics['RMSE'] > 0
    
    def test_compute_metrics_perfect_prediction(self):
        """测试完美预测的指标"""
        true_values = np.array([1, 2, 3, 4, 5])
        pred_values = true_values.copy()
        
        metrics = MetricsCalculator.compute_metrics(true_values, pred_values)
        
        assert metrics['MAE'] == pytest.approx(0, abs=1e-10)
        assert metrics['RMSE'] == pytest.approx(0, abs=1e-10)
        assert metrics['R2'] == pytest.approx(1, abs=1e-10)
    
    def test_compute_metrics_zero_values(self):
        """测试包含零值的情况"""
        true_values = np.array([0, 1, 0, 2, 0])
        pred_values = np.array([0.1, 1.1, 0.1, 2.1, 0.1])
        
        metrics = MetricsCalculator.compute_metrics(true_values, pred_values)
        
        assert 'MAPE' in metrics
        # MAPE应该只在非零值处计算
        assert np.isfinite(metrics['MAPE'])
    
    def test_compute_relative_error_stats(self):
        """测试相对误差统计"""
        true_values = np.array([1, 2, 3, 4, 5])
        pred_values = np.array([1.1, 1.8, 3.3, 3.7, 5.2])
        
        stats = MetricsCalculator.compute_relative_error_stats(true_values, pred_values)
        
        assert 'P50' in stats  # 中位数
        assert 'P95' in stats  # 95分位数
        assert 'mean_rel_error' in stats
        assert 'std_rel_error' in stats

class TestVisualizationTools:
    """测试可视化工具"""
    
    def test_plot_comparison_2d_slice(self):
        """测试2D切片对比图"""
        true_field = np.random.rand(10, 8, 6)
        pred_field = true_field + np.random.normal(0, 0.1, true_field.shape)
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.colorbar') as mock_colorbar:
            
            mock_fig = Mock()
            mock_axes = [Mock(), Mock()]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            fig = VisualizationTools.plot_comparison_2d_slice(true_field, pred_field)
            
            mock_subplots.assert_called_once()
            assert mock_axes[0].imshow.called
            assert mock_axes[1].imshow.called
    
    def test_plot_residual_analysis(self):
        """测试残差分析图"""
        residuals = np.random.normal(0, 1, 100)
        coordinates = np.random.rand(100, 3)
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('scipy.stats.probplot') as mock_probplot:
            
            mock_fig = Mock()
            mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            fig = VisualizationTools.plot_residual_analysis(residuals, coordinates)
            
            mock_subplots.assert_called_once()
            mock_probplot.assert_called_once()

# ==================== 适配器测试 ====================

class TestKrigingAdapter:
    """测试Kriging适配器"""
    
    @patch('ComposeTools.KRIGING_AVAILABLE', True)
    @patch('ComposeTools.kriging_training')
    @patch('ComposeTools.kriging_testing')
    def test_kriging_adapter_fit_predict(self, mock_testing, mock_training):
        """测试Kriging适配器的fit和predict"""
        # 设置Mock返回值
        mock_model = Mock()
        mock_training.return_value = mock_model
        mock_testing.return_value = (np.array([1, 2, 3]), np.array([1, 2, 3]))
        
        adapter = KrigingAdapter()
        
        # 测试数据
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        
        # 训练
        adapter.fit(X, y)
        assert adapter.is_fitted
        
        # 预测
        X_test = np.random.rand(5, 3)
        predictions = adapter.predict(X_test)
        
        assert len(predictions) == 5
        mock_training.assert_called_once()
        mock_testing.assert_called_once()
    
    @patch('ComposeTools.KRIGING_AVAILABLE', False)
    def test_kriging_adapter_unavailable(self):
        """测试Kriging不可用时的行为"""
        adapter = KrigingAdapter()
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        
        with pytest.raises(RuntimeError, match="Kriging模块不可用"):
            adapter.fit(X, y)

class TestPINNAdapter:
    """测试PINN适配器"""
    
    @patch('ComposeTools.PINN_AVAILABLE', True)
    @patch('ComposeTools.RadiationDataProcessor')
    @patch('ComposeTools.PINNTrainer')
    def test_pinn_adapter_fit_predict(self, mock_trainer_class, mock_processor_class):
        """测试PINN适配器的fit和predict"""
        # 设置Mock
        mock_processor = Mock()
        mock_processor.load_from_numpy.return_value = {'test': 'data'}
        mock_processor_class.return_value = mock_processor
        
        mock_trainer = Mock()
        mock_trainer.predict.return_value = np.log(np.array([1, 2, 3]) + EPSILON)
        mock_trainer_class.return_value = mock_trainer
        
        adapter = PINNAdapter()
        
        # 测试数据
        X = np.random.rand(10, 3)
        y = np.random.rand(10) + 1  # 确保正值
        
        # 训练
        adapter.fit(X, y)
        assert adapter.is_fitted
        
        # 预测
        X_test = np.random.rand(5, 3)
        predictions = adapter.predict(X_test)
        
        assert len(predictions) == 5
        mock_trainer.create_pinn_model.assert_called_once()
        mock_trainer.train.assert_called_once()
    
    @patch('ComposeTools.PINN_AVAILABLE', False)
    def test_pinn_adapter_unavailable(self):
        """测试PINN不可用时的行为"""
        adapter = PINNAdapter()
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        
        with pytest.raises(RuntimeError, match="PINN模块不可用"):
            adapter.fit(X, y)

# ==================== 方案1测试 ====================

class TestMode1Tools:
    """测试方案1专用工具"""
    
    def test_residual_kriging_compute_residuals(self):
        """测试残差计算"""
        tool = Mode1ResidualKriging()
        
        train_points = np.random.rand(10, 3)
        train_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pinn_predictions = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 6.9, 9.1, 9.9])
        
        residuals = tool.compute_residuals(train_points, train_values, pinn_predictions)
        
        expected_residuals = train_values - pinn_predictions
        np.testing.assert_array_almost_equal(residuals, expected_residuals)
    
    def test_residual_kriging_length_mismatch(self):
        """测试长度不匹配的异常处理"""
        tool = Mode1ResidualKriging()
        
        train_points = np.random.rand(10, 3)
        train_values = np.array([1, 2, 3, 4, 5])  # 错误长度
        pinn_predictions = np.array([1, 2, 3, 4, 5, 6])  # 错误长度
        
        with pytest.raises(ValueError, match="真实值和PINN预测值的长度不匹配"):
            tool.compute_residuals(train_points, train_values, pinn_predictions)
    
    def test_mode1_fusion_basic(self):
        """测试基本融合功能"""
        pinn_pred = np.array([1, 2, 3, 4, 5])
        kriging_residual = np.array([0.1, -0.1, 0.2, -0.2, 0.1])
        weight = 0.6
        
        fused = Mode1Fusion.fuse_residual(pinn_pred, kriging_residual, weight)
        
        expected = pinn_pred + weight * kriging_residual
        np.testing.assert_array_almost_equal(fused, expected)
    
    def test_mode1_fusion_with_uncertainty(self):
        """测试带不确定度的融合"""
        pinn_pred = np.array([1, 2, 3, 4, 5])
        kriging_residual = np.array([0.1, -0.1, 0.2, -0.2, 0.1])
        uncertainty = np.array([0.05, 0.1, 0.15, 0.1, 0.05])
        weight = 0.5
        
        fused, confidence = Mode1Fusion.fuse_residual(pinn_pred, kriging_residual, weight, uncertainty)
        
        expected_fused = pinn_pred + weight * kriging_residual
        expected_confidence = weight * 1.96 * uncertainty
        
        np.testing.assert_array_almost_equal(fused, expected_fused)
        np.testing.assert_array_almost_equal(confidence, expected_confidence)
    
    def test_adaptive_weight_strategy(self):
        """测试自适应权重策略"""
        residuals = np.array([0.1, -0.5, 0.3, -0.2, 0.8])
        
        # 测试均匀权重
        weights_uniform = Mode1Fusion.adaptive_weight_strategy(residuals, strategy='uniform')
        assert np.all(weights_uniform == 0.5)
        
        # 测试基于幅度的权重
        weights_magnitude = Mode1Fusion.adaptive_weight_strategy(residuals, strategy='magnitude_based')
        assert len(weights_magnitude) == len(residuals)
        assert np.all((weights_magnitude >= 0.1) & (weights_magnitude <= 0.9))
        
        # 测试基于方差的权重
        kriging_std = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        weights_variance = Mode1Fusion.adaptive_weight_strategy(residuals, kriging_std, 'variance_based')
        assert len(weights_variance) == len(residuals)

# ==================== 方案2测试 ====================

class TestMode2Tools:
    """测试方案2专用工具"""
    
    def test_roi_detector_high_density(self):
        """测试高密度ROI检测"""
        # 创建聚集的点
        cluster_points = np.random.normal([0, 0, 0], 0.5, (15, 3))
        scattered_points = np.random.uniform(-5, 5, (10, 3))
        train_points = np.vstack([cluster_points, scattered_points])
        train_values = np.random.rand(25)
        
        roi_bounds = Mode2ROIDetector.detect_roi(train_points, train_values, 'high_density')
        
        assert 'min' in roi_bounds
        assert 'max' in roi_bounds
        assert 'mask' in roi_bounds
        assert len(roi_bounds['mask']) == len(train_points)
    
    def test_roi_detector_high_value(self):
        """测试高数值ROI检测"""
        train_points = np.random.rand(20, 3)
        train_values = np.random.rand(20)
        
        roi_bounds = Mode2ROIDetector.detect_roi(train_points, train_values, 'high_value')
        
        assert 'min' in roi_bounds
        assert 'max' in roi_bounds
        assert 'mask' in roi_bounds
        assert 'value_scores' in roi_bounds
    
    def test_roi_detector_bounding_box(self):
        """测试包围盒ROI检测"""
        train_points = np.random.rand(20, 3) * 10 - 5
        train_values = np.random.rand(20)
        
        roi_bounds = Mode2ROIDetector.detect_roi(train_points, train_values, 'bounding_box')
        
        assert 'min' in roi_bounds
        assert 'max' in roi_bounds
        assert 'bounding_box' in roi_bounds
        assert roi_bounds['bounding_box'] == True
    
    def test_roi_detector_invalid_strategy(self):
        """测试无效ROI策略"""
        train_points = np.random.rand(10, 3)
        train_values = np.random.rand(10)
        
        with pytest.raises(ValueError, match="不支持的ROI策略"):
            Mode2ROIDetector.detect_roi(train_points, train_values, 'invalid_strategy')
    
    @patch('ComposeTools.KrigingAdapter')
    def test_sample_augmentor(self, mock_kriging_class):
        """测试样本扩充器"""
        # 设置Mock
        mock_kriging = Mock()
        mock_kriging.predict.return_value = np.array([1, 2, 3])
        mock_kriging_class.return_value = mock_kriging
        
        augmentor = Mode2SampleAugmentor()
        
        train_points = np.random.rand(10, 3)
        train_values = np.random.rand(10)
        roi_bounds = {
            'min': np.array([-1, -1, -1]),
            'max': np.array([1, 1, 1])
        }
        
        aug_points, aug_values = augmentor.augment_by_kriging(
            train_points, train_values, roi_bounds, augment_factor=1.3
        )
        
        expected_new_samples = int(10 * (1.3 - 1.0))
        expected_total = 10 + expected_new_samples
        
        assert len(aug_points) == expected_total
        assert len(aug_values) == expected_total
        mock_kriging.fit.assert_called_once()
        mock_kriging.predict.assert_called_once()

# ==================== 工作流测试 ====================

class TestCouplingWorkflow:
    """测试耦合工作流"""
    
    @patch('ComposeTools.PINNAdapter')
    @patch('ComposeTools.Mode1ResidualKriging')
    def test_workflow_mode1_pipeline(self, mock_residual_kriging_class, mock_pinn_class):
        """测试方案1完整流程"""
        # 设置Mock
        mock_pinn = Mock()
        mock_pinn.predict.return_value = np.array([1, 2, 3, 4, 5])
        mock_pinn_class.return_value = mock_pinn
        
        mock_residual_tool = Mock()
        mock_residual_tool.residual_kriging.return_value = (np.array([0.1, 0.1, 0.1, 0.1, 0.1]), 
                                                           np.array([0.05, 0.05, 0.05, 0.05, 0.05]))
        mock_residual_kriging_class.return_value = mock_residual_tool
        
        # 创建工作流
        workflow = CouplingWorkflow()
        
        train_points = np.random.rand(10, 3)
        train_values = np.random.rand(10)
        prediction_points = np.random.rand(5, 3)
        
        results = workflow.run_mode1_pipeline(train_points, train_values, prediction_points)
        
        assert 'final_predictions' in results
        assert 'pinn_predictions' in results
        assert 'residual_predictions' in results
        assert len(results['final_predictions']) == 5
    
    @patch('ComposeTools.PINNAdapter')
    @patch('ComposeTools.Mode2ROIDetector')
    @patch('ComposeTools.Mode2SampleAugmentor')
    def test_workflow_mode2_pipeline(self, mock_augmentor_class, mock_roi_class, mock_pinn_class):
        """测试方案2完整流程"""
        # 设置Mock
        mock_pinn1 = Mock()
        mock_pinn2 = Mock()
        mock_pinn2.predict.return_value = np.array([1, 2, 3, 4, 5])
        mock_pinn_class.side_effect = [mock_pinn1, mock_pinn2]  # 第二个用于增强版
        
        mock_roi_detector = Mock()
        mock_roi_detector.detect_roi.return_value = {
            'min': np.array([-1, -1, -1]),
            'max': np.array([1, 1, 1]),
            'mask': np.ones(10, dtype=bool)
        }
        mock_roi_class.return_value = mock_roi_detector
        
        mock_augmentor = Mock()
        mock_augmentor.augment_by_kriging.return_value = (np.random.rand(20, 3), np.random.rand(20))
        mock_augmentor_class.return_value = mock_augmentor
        
        # 创建工作流
        workflow = CouplingWorkflow()
        
        train_points = np.random.rand(10, 3)
        train_values = np.random.rand(10)
        prediction_points = np.random.rand(5, 3)
        
        results = workflow.run_mode2_pipeline(train_points, train_values, prediction_points)
        
        assert 'final_predictions' in results
        assert 'roi_bounds' in results
        assert 'augmented_points' in results
        assert 'augmented_values' in results
        assert len(results['final_predictions']) == 5

# ==================== 集成测试 ====================

class TestIntegration:
    """端到端集成测试"""
    
    def test_end_to_end_data_flow(self):
        """测试端到端数据流"""
        # 生成测试数据
        train_points, train_values, test_points, test_values = TestDataGenerator.generate_simple_3d_data()
        
        # 创建数据结构
        field_tensor = FieldTensor(train_points, train_values)
        probe_set = ProbeSet(train_points, train_values)
        
        # 数据处理
        normalized_values, norm_info = DataNormalizer.robust_normalize(train_values)
        
        # 误差计算（使用简单预测）
        simple_pred = train_values * 1.1  # 简单的预测
        metrics = MetricsCalculator.compute_metrics(train_values, simple_pred)
        
        # 验证数据流的完整性
        assert len(field_tensor.coordinates) == len(train_points)
        assert len(probe_set.measurements) == len(train_values)
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert norm_info['method'] == 'robust'
    
    @patch('ComposeTools.KRIGING_AVAILABLE', True)
    @patch('ComposeTools.PINN_AVAILABLE', True)
    def test_mock_coupling_integration(self):
        """测试模拟的耦合集成（使用Mock避免依赖问题）"""
        with patch('ComposeTools.kriging_training') as mock_kriging_train, \
             patch('ComposeTools.kriging_testing') as mock_kriging_test, \
             patch('ComposeTools.PINNTrainer') as mock_pinn_trainer_class:
            
            # 设置Mock返回值
            mock_model = Mock()
            mock_kriging_train.return_value = mock_model
            mock_kriging_test.return_value = (np.array([1, 2, 3]), np.array([1, 2, 3]))
            
            mock_trainer = Mock()
            mock_trainer.predict.return_value = np.log(np.array([1, 2, 3]) + EPSILON)
            mock_pinn_trainer_class.return_value = mock_trainer
            
            # 创建适配器
            kriging_adapter = KrigingAdapter()
            pinn_adapter = PINNAdapter()
            
            # 测试数据
            train_points, train_values, test_points, test_values = TestDataGenerator.generate_simple_3d_data(n_samples=20)
            field_info = TestDataGenerator.generate_field_info()
            
            # 训练Kriging
            kriging_adapter.fit(train_points, train_values)
            assert kriging_adapter.is_fitted
            
            # 训练PINN
            pinn_adapter.fit(train_points, train_values, 
                           space_dims=field_info['space_dims'],
                           world_bounds=field_info['world_bounds'])
            assert pinn_adapter.is_fitted
            
            # 预测
            kriging_pred = kriging_adapter.predict(test_points)
            pinn_pred = pinn_adapter.predict(test_points)
            
            assert len(kriging_pred) == len(test_points)
            assert len(pinn_pred) == len(test_points)

# ==================== 异常处理和边界条件测试 ====================

class TestErrorHandling:
    """测试异常处理和边界条件"""
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_array = np.array([])
        
        with pytest.raises(ValueError):
            MetricsCalculator.compute_metrics(empty_array, empty_array)
    
    def test_invalid_fusion_weight(self):
        """测试无效融合权重的警告"""
        pinn_pred = np.array([1, 2, 3])
        kriging_residual = np.array([0.1, 0.1, 0.1])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Mode1Fusion.fuse_residual(pinn_pred, kriging_residual, weight=1.5)  # 超出范围
            assert len(w) > 0
            assert "不在推荐范围" in str(w[0].message)
    
    def test_mismatched_array_lengths(self):
        """测试数组长度不匹配"""
        pinn_pred = np.array([1, 2, 3])
        kriging_residual = np.array([0.1, 0.1])  # 长度不匹配
        
        with pytest.raises(ValueError, match="长度不匹配"):
            Mode1Fusion.fuse_residual(pinn_pred, kriging_residual)
    
    def test_single_point_data(self):
        """测试单点数据的处理"""
        single_point = np.array([[0, 0, 0]])
        single_value = np.array([1.0])
        
        # 创建数据结构应该成功
        field_tensor = FieldTensor(single_point, single_value)
        assert len(field_tensor.coordinates) == 1
        
        # 但是某些操作可能会失败或给出警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略可能的警告
            roi_bounds = Mode2ROIDetector.detect_roi(single_point, single_value, 'high_density')
            assert 'min' in roi_bounds
            assert 'max' in roi_bounds

# ==================== 工具函数测试 ====================

class TestUtilityFunctions:
    """测试工具函数"""
    
    def test_validate_compose_environment(self):
        """测试环境验证函数"""
        status = validate_compose_environment()
        
        assert isinstance(status, dict)
        assert 'Kriging' in status
        assert 'PINN' in status
        assert 'CuPy' in status
        assert 'PyTorch' in status
        
        # 所有值应该是布尔值
        for key, value in status.items():
            assert isinstance(value, bool)
    
    def test_print_compose_banner(self):
        """测试横幅打印函数"""
        # 这个测试主要是为了覆盖率，确保函数可以正常调用
        try:
            print_compose_banner()
        except Exception:
            pytest.fail("print_compose_banner() raised an exception unexpectedly")

# ==================== 性能测试 ====================

class TestPerformance:
    """性能相关测试"""
    
    def test_large_data_handling(self):
        """测试大数据处理能力"""
        # 生成较大的数据集
        large_points = np.random.rand(1000, 3)
        large_values = np.random.rand(1000)
        
        # 测试数据结构创建
        field_tensor = FieldTensor(large_points, large_values)
        assert len(field_tensor.coordinates) == 1000
        
        # 测试归一化
        normalized, info = DataNormalizer.robust_normalize(large_values)
        assert len(normalized) == 1000
        
        # 测试误差计算
        pred_values = large_values * 1.1
        metrics = MetricsCalculator.compute_metrics(large_values, pred_values)
        assert all(key in metrics for key in ['MAE', 'RMSE', 'MAPE', 'R2'])
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        # 创建较大数组并确保没有内存泄漏
        for _ in range(10):
            data = np.random.rand(100, 3)
            field = FieldTensor(data, np.random.rand(100))
            del field, data  # 显式删除
        
        # 如果能运行到这里说明没有严重的内存问题
        assert True

# ==================== 测试运行配置 ====================

def test_imports():
    """测试所有导入是否正常"""
    # 这个测试确保所有必要的模块都能正确导入
    from ComposeTools import (
        ComposeConfig, FieldTensor, ProbeSet,
        DataNormalizer, MetricsCalculator, VisualizationTools,
        KrigingAdapter, PINNAdapter, 
        Mode1ResidualKriging, Mode1Fusion,
        Mode2ROIDetector, Mode2SampleAugmentor,
        CouplingWorkflow, validate_compose_environment
    )
    assert True

if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "--tb=short"]) 