# 项目参数优化待办清单 (Parameters TODO List)

本文件旨在记录和追踪`ComposeProject`中可以进一步优化和开放的参数，以提升代码的灵活性和规范性。

## 一、 已有但需明确的可选参数

这些参数功能已实现，但建议在文档和接口中更加明确地说明其可配置性。

- [x] **融合权重 (`fusion_weight`)**: `模式1`中PINN与Kriging残差的融合比例。
- [x] **Kriging变异函数 (`kriging_variogram_model`)**: `模式1`和`模式2`中Kriging插值的模型。
- [x] **ROI检测策略 (`roi_detection_strategy`)**: `模式2`中用于识别关键区域的算法。
- [x] **样本扩充倍数 (`augment_factor`)**: `模式2`中在ROI内生成合成样本的比例。
- [x] **PINN训练参数 (`epochs`, `loss_weights`等)**: `模式1`和`模式2`中控制神经网络训练的超参数。

## 二、 建议从硬编码改为可选参数

这些参数目前在代码中以固定值（硬编码）存在，将其开放为可选参数将极大提升工具的灵活性和可扩展性。

### 模式1 待办

- [ ] **开放融合策略 (`fusion_strategy`)**
    - **当前状态**: 仅支持静态加权融合 (`pinn + weight * residual`)。
    - **改进建议**: 增加一个`fusion_strategy`参数，允许用户选择不同的融合算法（例如， `'static'`, `'variance_based'`），以支持更复杂的实验，特别是调用已有的`adaptive_weight_strategy`方法。

### 模式2 待办

- [ ] **开放ROI检测阈值 (`roi_strategy_params`)**
    - **当前状态**: ROI检测策略的内部参数是硬编码的，例如 `density_percentile=75`。
    - **改进建议**: 允许用户通过一个字典（如`strategy_params`）向工作流传递这些具体阈值，使得ROI的定义能够灵活适应不同数据的分布特性。

- [ ] **开放数据增强的采样策略 (`augment_sampling_strategy`)**
    - **当前状态**: 在ROI内生成新样本点的策略被硬编码为`'grid'`（网格采样）。
    - **改进建议**: 将`sampling_strategy`参数暴露给用户，允许选择`'grid'`, `'random'`等不同的采样方式，以探索不同数据增强分布对PINN模型重训练的影响。 