# PINN-Kriging 耦合系统配置指南

## 🚀 快速开始

### 基本用法
```bash
# 使用默认配置
python main.py

# 选择预设
python main.py --preset quick_test          # 小规模快速跑通
python main.py --preset kriging_only        # 强制 Kriging 流程
python main.py --preset pinn_only           # 强制 PINN 流程
python main.py --preset random_sampling     # 随机采样版本（向后兼容）
python main.py --preset adaptive_experiment_config1  # 自适应实验示例（旧版仅数据注入复现）

# 强制指定方法（覆盖预设中的 system.method）
python main.py --method kriging
python main.py --method pinn
python main.py --method adaptive_experiment
python main.py --method auto                # 智能选择（默认）
```

## ⚙️ 预测方法选择（配置示例）
```python
@dataclass
class SystemConfig:
    method: str = "auto"                 # auto/kriging/pinn/adaptive_experiment
    enable_pinn_adaptive: bool = False
    enable_data_injection: bool = False  # pinn 使用；adaptive_experiment 在专属段配置
```
用途速览：
- auto：自动选择 Kriging 或 PINN
- kriging：纯 Kriging 插值
- pinn：纯 PINN（可选随机自适应加密）
- adaptive_experiment：多周期 + 数据注入 + Kriging 重采样 + 基线对比

### 智能选择规则（AutoSelectionWorkflow）
- 点数不足 `selection.min_points_for_kriging`（默认100）→ PINN
- 最近邻CV < `uniformity_cv_threshold`（默认0.6）→ Kriging；否则 PINN

## 📋 配置预设
当前代码内置以下预设（`config.py -> PRESETS`）：

### 1. `quick_test` - 快速测试
```bash
python main.py --preset quick_test
```
- 🎯 **用途**: 快速验证系统功能
- 📊 **数据规模**: 50 个训练样本（采样区域缩小，步长10）
- 🧠 **PINN训练**: `total_epochs=1000`，`detect_every=200`，`adaptive_cycle_epochs=500`，collocation 1024
- ⚙️ **方法**: system.method=auto

### 2. `kriging_only` - 仅克里金
```bash
python main.py --preset kriging_only
```
- 🎯 **用途**: 强制走 Kriging 工作流
- ⚙️ **选择规则**: `min_points_for_kriging=1`、`uniformity_cv_threshold=999`，保证决策为 Kriging
- 📊 **采样**: kriging_style，步长5

### 3. `pinn_only` - 仅 PINN
```bash
python main.py --preset pinn_only
```
- 🎯 **用途**: 强制走 PINN 工作流
- ⚙️ **选择规则**: `min_points_for_kriging=99999`，保证决策为 PINN
- 📊 **采样**: kriging_style，步长5

### 4. `random_sampling` - 随机采样向后兼容
```bash
python main.py --preset random_sampling
```
- 🎯 **用途**: 采用随机采样（strategy=positive_only），保持旧版调用兼容
- 📊 **采样**: `random_sampling.num_samples=300`
- ⚙️ **方法**: system.method=auto

### 5. `adaptive_experiment_config1` - 自适应实验示例（旧版仅数据注入复现）
```bash
python main.py --preset adaptive_experiment_config1 --method adaptive_experiment
```
- 🎯 **用途**: 复现旧版自适应（无 Kriging，仅注入）
- 🧠 **自适应**: `total_epochs=4000`，`adaptive_cycle_epochs=400`，`enable_kriging=False`，`enable_data_injection=True`
- 📊 **数据**: 训练样本 50，测试集 300

> 说明：未指定 `--method` 时采用预设中的 `system.method`；CLI 指定则覆盖。

## 🔧 自定义配置（示例结构）
```python
@dataclass
class DataConfig:
    num_samples: int = 200
    test_set_size: int = 30000
    space_dims: List[float] = field(default_factory=lambda:[20.0,10.0,10.0])

@dataclass
class PinnConfig:
    network_layers: List[int] = field(default_factory=lambda:[3,64,64,64,1])
    num_collocation_points: int = 4096
    learning_rate: float = 1e-3
    loss_ratio: float = 10.0
    total_epochs: int = 5000
    detect_every: int = 100
    adaptive_cycle_epochs: int = 2000
    detection_threshold: float = 0.2

@dataclass
class KrigingConfig:
    variogram_model: str = "exponential"
    nlags: int = 8
    block_size: int = 10000
    exploration_ratio: float = 0.2
    total_candidates: int = 50000
    style: str = "gpu_b"
    multi_process: bool = False
    print_time: bool = False
    torch_ac: bool = False

@dataclass
class AdaptiveExperimentConfig:
    total_epochs: int = 2000
    adaptive_cycle_epochs: int = 200
    detect_every: int = 100
    num_residual_scout_points: int = 5000
    exploration_initial: float = 0.5
    exploration_final: float = 0.018
    exploration_decay: float = 0.04
    enable_kriging: bool = False
    enable_data_injection: bool = True
    enable_rapid_improvement_early_stop: bool = True
    split_ratios: list = field(default_factory=lambda:[0.7,0.05,0.05,0.05,0.05,0.05,0.05])
    test_set_size: int = 30000
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
- **插值对比**: 使用 `kriging_only` / `pinn_only`
- **自适应实验**: 使用 `adaptive_experiment_config1` 或自行配置 `adaptive_experiment`

### Q: 如何强制使用特定方法？
```bash
# 强制使用Kriging（适合均匀数据）
python main.py --method kriging

# 强制使用PINN（适合复杂数据）
python main.py --method pinn
```

### Q: 如何调整训练时间？
修改 `config.py` 中的 `total_epochs` / `cycle_epochs` / `adaptive_cycle_epochs`：
- 快速测试: 1000轮
- 标准训练: 4000轮（示例，可自定）
- 高精度: 8000轮（示例，可自定）

### Q: 如何增加数据规模？
修改 `config.py` 中的 `num_samples` 和 `test_set_size`:
```python
num_samples: int = 500        # 训练样本
test_set_size: int = 1000     # 测试样本
```

## 🎯 推荐工作流

1. **初次使用**: `python main.py --preset quick_test`
2. **验证功能**: `python main.py --method kriging` 和 `python main.py --method pinn`
3. **自适应实验**: `python main.py --method adaptive_experiment`（可用 `adaptive_experiment_config1` 预设或自定义配置）
4. **自定义配置**: 修改 `config.py` 后运行 `python main.py`

## 📞 技术支持

如有问题，请检查：
1. 依赖环境是否正确安装
2. 配置文件语法是否正确
3. 使用 `--verbose` 参数查看详细错误信息 