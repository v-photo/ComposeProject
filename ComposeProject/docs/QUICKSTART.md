# 快速入门指南 (Quick Start Guide)

本文档指导新手快速上手运行 PINN-Kriging 耦合系统。

---

## 一、项目概述

本项目是一个 **GPU加速的 Kriging × PINN 耦合重建系统**，用于三维辐射场重建。系统会根据数据分布特征自动选择最优的预测方法（Kriging 或 PINN）。

---

## 二、项目入口文件

| 入口文件 | 路径 | 用途 |
|---------|------|------|
| **主入口** | `ComposeProject/main.py` | 运行完整的耦合工作流 |
| PINN独立测试 | `PINN/PINNTest.ipynb` | Jupyter Notebook 测试PINN |
| Kriging独立测试 | `Kriging/main.ipynb` | Jupyter Notebook 测试Kriging |
| 数据对比工具 | `compare_data.py` | 对比两种数据处理流程的一致性 |

---

## 三、环境准备

### 3.1 依赖安装

确保已安装以下依赖：

```bash
# 核心依赖
pip install numpy pandas scikit-learn deepxde torch

# GPU加速（可选但推荐）
pip install cupy-cuda11x  # 根据你的CUDA版本选择

# Kriging依赖
pip install pykrige
```

### 3.2 数据文件

默认数据文件位置：`PINN/DATA.xlsx`

数据格式要求：
- Excel文件，包含多个Sheet（每个Sheet对应一个Z层）
- Sheet命名格式：`avg_1_1`, `avg_1_2`, ..., `avg_1_72`
- 每个Sheet包含 Y×X 的剂量网格数据

---

## 四、运行项目

### 4.1 基本运行（使用默认配置）

```bash
cd ComposeProject
python main.py
```

### 4.2 使用预设配置

项目提供了多种预设配置，通过 `--preset` 参数选择：

```bash
# 快速测试（1000轮训练，约5秒）
python main.py --preset quick_test

# 默认配置（5000轮训练）
python main.py --preset default

# 强制使用Kriging方法
python main.py --preset kriging_only

# 强制使用PINN方法
python main.py --preset pinn_only
```

### 4.3 预设配置详解

| 预设名称 | 说明 | 训练轮数 | 适用场景 |
|---------|------|----------|---------|
| `default` | 默认配置 | 5000 | 日常使用 |
| `quick_test` | 快速测试 | 1000 | 验证环境、调试代码 |
| `kriging_only` | 强制Kriging | - | 数据分布均匀时 |
| `pinn_only` | 强制PINN | 5000 | 数据分布不均匀时 |

---

## 五、配置参数

### 5.1 配置文件位置

所有配置集中在 `ComposeProject/config.py` 文件中。

### 5.2 主要配置项

```python
DEFAULT_CONFIG = {
    "experiment": {
        "name": "default_adaptive_pinn",  # 实验名称
    },
    "data": {
        "file_path": "../PINN/DATA.xlsx",  # 数据文件路径
        "num_samples": 300,                 # 训练采样点数
        "downsample_factor": 1,             # 降采样系数（1=不降采样）
    },
    "pinn": {
        "model_params": {
            "network_layers": [3, 64, 64, 64, 1],  # 网络结构
            "learning_rate": 1e-3,                  # 学习率
            "num_collocation_points": 4096,         # 配置点数量
        },
        "training_params": {
            "total_epochs": 5000,    # 总训练轮数
            "detect_every": 500,     # 每N轮检测一次
        },
    },
    "kriging": {
        "variogram_model": "exponential",  # 变异函数模型
        "nlags": 8,                         # 距离分组数
        "block_size": 10000,                # GPU分块大小
    },
    "selection": {
        "min_points_for_kriging": 100,     # Kriging最少需要的点数
        "uniformity_cv_threshold": 0.6,    # 均匀性阈值（CV值）
    },
    "system": {
        "use_gpu": True,          # 是否使用GPU
        "random_seed": 42,        # 随机种子
        "verbose": True,          # 详细输出
    }
}
```

### 5.3 自定义配置

1. 打开 `config.py`
2. 在 `PRESETS` 字典中添加新预设，或修改 `DEFAULT_CONFIG`
3. 运行时使用 `--preset <预设名>` 选择

---

## 六、输出说明

### 6.1 控制台输出

程序运行时会输出以下信息：

1. **依赖状态检查**：显示 Kriging、CuPy、PyTorch 等模块是否可用
2. **配置信息**：当前使用的完整配置
3. **数据加载**：3D数据加载进度和统计信息
4. **方法选择**：自动选择的预测方法（Kriging 或 PINN）
5. **训练进度**：实时显示损失值和MRE指标
6. **结果统计**：预测结果的统计信息

### 6.2 输出文件

- `results/training_history_<实验名>.png`：训练历史曲线图
- `models/pinn_checkpoint-<轮数>.pt`：模型检查点

---

## 七、常见问题

### Q1: 如何加速调试？

使用 `quick_test` 预设：
```bash
python main.py --preset quick_test
```

或修改配置降低训练轮数：
```python
"training_params": {"total_epochs": 500}
```

### Q2: GPU内存不足怎么办？

1. 减少配置点数量：
```python
"num_collocation_points": 1024  # 默认4096
```

2. 减少训练样本数：
```python
"num_samples": 100  # 默认300
```

### Q3: 如何更换数据文件？

修改 `config.py` 中的 `data.file_path`：
```python
"file_path": "/path/to/your/data.xlsx"
```

### Q4: 程序一直选择PINN而不是Kriging？

检查 `selection.uniformity_cv_threshold` 参数。降低此值会更倾向于选择PINN：
```python
"uniformity_cv_threshold": 0.8  # 提高此值更倾向于Kriging
```

---

## 八、下一步

- 阅读 `TESTING_GUIDE.md` 了解如何单独测试PINN和Kriging
- 查看 `README_CONFIG.md` 了解更多配置选项
- 查看 `src/` 目录了解代码架构

---

*最后更新: 2024年*
