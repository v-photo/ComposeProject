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

# 智能选择方法（默认）
python main.py --method auto
```

## ⚙️ 预测方法选择

系统支持三种预测方法选择模式：

| 参数 | 说明 | 适用场景 |
|------|------|----------|
| `--method auto` | 🤖 **智能选择**（默认）| 系统自动分析数据分布，选择最适合的方法 |
| `--method kriging` | ⚙️ **强制Kriging** | 数据分布均匀，需要空间插值 |
| `--method pinn` | 🧠 **强制PINN** | 数据稀疏或分布不均，需要物理约束 |

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

## 🔧 自定义配置

### 修改 config.py

打开 `config.py` 文件，可以自定义以下配置：

#### 实验配置 (ExperimentConfig)
```python
@dataclass
class ExperimentConfig:
    experiment_name: str = "my_experiment"      # 实验名称
    enable_kriging_resampling: bool = True      # 启用克里金重采样
    enable_data_injection: bool = False         # 启用数据注入
    enable_rapid_improvement_early_stop: bool = False  # 启用快速改善早停
```

#### 数据配置 (DataConfig)
```python
@dataclass  
class DataConfig:
    num_samples: int = 100                      # 训练样本数量
    test_set_size: int = 200                    # 测试集大小
    space_dims: List[float] = [20.0, 10.0, 10.0]  # 物理空间尺寸 [x,y,z] (米)
```

#### PINN配置 (PINNConfig)
```python
@dataclass
class PINNConfig:
    total_epochs: int = 4000                    # 总训练轮数
    network_layers: List[int] = [3, 64, 64, 64, 1]  # 网络结构
    num_collocation_points: int = 4096          # 配置点数量
    initial_loss_ratio: float = 10.0           # 初始损失权重比值
    final_loss_ratio: float = 0.1              # 最终损失权重比值
```

#### 克里金配置 (KrigingConfig)
```python
@dataclass
class KrigingConfig:
    variogram_model: str = "exponential"        # 变异函数模型
    initial_exploration_ratio: float = 0.50    # 初始探索率
    final_exploration_ratio: float = 0.18      # 最终探索率
    exploration_decay_rate: float = 0.03       # 探索率衰减
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