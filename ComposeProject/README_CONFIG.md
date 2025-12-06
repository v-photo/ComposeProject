# PINN-Kriging 耦合系统配置指南

## 🚀 快速开始

### 基本用法
```bash
# 使用默认配置
python main.py

# 使用预设配置进行快速测试
python main.py --preset quick_test

# 强制使用Kriging方法
python main.py --method kriging

# 强制使用PINN方法  
python main.py --method pinn

# 使用 Compose（PINN + GPU-Kriging 引导）
python main.py --method compose

# 运行自适应完整实验（周期训练+数据注入+Kriging）
python main.py --method adaptive_experiment

# 智能选择方法（默认）
python main.py --method auto
```

## ⚙️ 预测方法选择（配置示例）
```python
@dataclass
class SystemConfig:
    method: str = "auto"                 # auto/kriging/pinn/compose/adaptive_experiment
    enable_compose_adaptive: bool = False
    enable_pinn_adaptive: bool = False
    enable_data_injection: bool = False  # 仅 compose/pinn 使用；adaptive_experiment 独立配置
```
用途速览：
- auto：自动选择 Kriging 或 PINN
- kriging：纯 Kriging 插值
- pinn：纯 PINN（可选随机自适应加密）
- compose：两阶段 PINN + Kriging 残差引导
- adaptive_experiment：多周期 + 数据注入 + Kriging 重采样 + 基线对比

### 智能选择规则
- **数据分布均匀** + **样本充足** → 自动选择 Kriging
- **数据聚集** 或 **样本稀少** → 自动选择 PINN

## 📋 配置预设
使用 `--preset` 参数选择预设配置：

### 1. `quick_test` - 快速测试
```bash
python main.py --preset quick_test
```
- 🎯 **用途**: 快速验证系统功能
- ⏱️ **训练时间**: ~5秒
- 📊 **数据规模**: 50个训练样本，100个测试样本
- 🧠 **PINN训练**: 1000轮（简化）

### 2. `full_adaptive` - 完整自适应训练
```bash
python main.py --preset full_adaptive
```
- 🎯 **用途**: 生产环境，最佳性能
- ⏱️ **训练时间**: ~30-60秒
- 📊 **数据规模**: 200个训练样本，500个测试样本
- 🧠 **PINN训练**: 8000轮，包含克里金重采样和数据注入
> 若要使用新增自适应实验，请配合 `--method adaptive_experiment`（或在 preset 中将 `system.method` 设为 `adaptive_experiment`），并在 `config.py -> adaptive_experiment` 调整周期、探索率、注入/Kriging 开关等。

### 3. `kriging_only` - 仅克里金重采样
```bash
python main.py --preset kriging_only
```
- 🎯 **用途**: 测试克里金重采样效果
- ⏱️ **训练时间**: ~20-40秒
- 📊 **数据规模**: 150个训练样本，300个测试样本
- 🧠 **PINN训练**: 6000轮，启用克里金重采样

### 4. `baseline` - 基线对比
```bash
python main.py --preset baseline
```
- 🎯 **用途**: 性能基线，不使用自适应策略
- ⏱️ **训练时间**: ~15-30秒
- 📊 **数据规模**: 100个训练样本，200个测试样本
- 🧠 **PINN训练**: 4000轮，固定损失权重

> 说明：各预设已在 `config.py` 的 `system.method` 设置默认方法（如 kriging_only→kriging，pinn_only→pinn，default/quick_test/random_sampling→auto）。未指定 `--method` 时采用预设默认，CLI 指定则覆盖。

## 🔧 自定义配置（示例结构）
```python
@dataclass
class DataConfig:
    num_samples: int = 300
    test_set_size: int = 300
    space_dims: List[float] = field(default_factory=lambda:[20.0,10.0,10.0])

@dataclass
class PinnConfig:
    network_layers: List[int] = field(default_factory=lambda:[3,64,64,64,1])
    num_collocation_points: int = 4096
    learning_rate: float = 1e-3
    loss_ratio: float = 10.0
    total_epochs: int = 5000
    detect_every: int = 500
    adaptive_cycle_epochs: int = 2000
    detection_threshold: float = 0.1

@dataclass
class KrigingConfig:
    variogram_model: str = "exponential"
    nlags: int = 8
    block_size: int = 10000
    exploration_ratio: float = 0.2        # compose 模式使用
    total_candidates: int = 50000         # compose 模式使用
    style: str = "gpu_b"
    multi_process: bool = False
    print_time: bool = False
    torch_ac: bool = False

@dataclass
class AdaptiveExperimentConfig:
    total_epochs: int = 1000
    adaptive_cycle_epochs: int = 200
    detect_every: int = 100
    num_residual_scout_points: int = 5000
    exploration_initial: float = 0.2
    exploration_final: float = 0.05
    exploration_decay: float = 0.02
    enable_kriging: bool = True
    enable_data_injection: bool = False
    enable_rapid_improvement_early_stop: bool = True
    split_ratios: list = field(default_factory=lambda:[0.7,0.05,0.05,0.05,0.05,0.05,0.05])
    test_set_size: int = 300
    enable_baseline: bool = True
    file_suffix: str = "full_adaptive"
```

## 📊 输出结果

### 控制台输出
- 🔍 **数据分布分析**: 显示数据均匀性和推荐方法
- 📈 **训练进度**: 实时显示损失值和测试指标
- 📋 **结果摘要**: 预测范围、耗时、使用方法

### 保存文件
- `results/predictions_<experiment_name>.npy`: 预测结果
- `results/training_history_<experiment_name>.npz`: 训练历史（如果使用PINN）


## 🛠️ 常见问题

### Q: 如何选择合适的预设？
- **新手**: 使用 `quick_test` 快速了解系统
- **研究**: 使用 `full_adaptive` 获得最佳性能
- **对比**: 使用 `baseline` 作为性能基准

### Q: 如何强制使用特定方法？
```bash
# 强制使用Kriging（适合均匀数据）
python main.py --method kriging

# 强制使用PINN（适合复杂数据）
python main.py --method pinn
```

### Q: 如何调整训练时间？
修改 `config.py` 中的 `total_epochs`:
- 快速测试: 1000轮 (~5秒)
- 标准训练: 4000轮 (~20秒)  
- 高精度: 8000轮 (~60秒)

### Q: 如何增加数据规模？
修改 `config.py` 中的 `num_samples` 和 `test_set_size`:
```python
num_samples: int = 500        # 增加到500个训练样本
test_set_size: int = 1000     # 增加到1000个测试样本
```

## 🎯 推荐工作流

1. **初次使用**: `python main.py --preset quick_test`
2. **验证功能**: `python main.py --method kriging` 和 `python main.py --method pinn`
3. **性能测试**: `python main.py --preset full_adaptive`
4. **自定义配置**: 修改 `config.py` 后运行 `python main.py`

## 📞 技术支持

如有问题，请检查：
1. 依赖环境是否正确安装
2. 配置文件语法是否正确
3. 使用 `--verbose` 参数查看详细错误信息 