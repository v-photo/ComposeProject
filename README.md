# GPU-Accelerated Block Kriging × PINN 耦合重建项目

**项目名称**: GPU-Accelerated Block Kriging & PINN for 3-D Radiation Field Reconstruction

## 核心特色

* **速度**: PyKrige 3D 普通克里金模块改写为 CUDA 分块实现，百倍级加速
* **精度**: 采用 Physics-Informed Neural Network (PINN) 融合少量测点与物理先验
* **耦合**: 提供两种 Kriging × PINN 深度融合模式，兼顾全局趋势与局部细节

## 项目结构

```
├── Kriging/                    # 原有Kriging模块
│   ├── myKriging.py           # Kriging主接口
│   ├── myPyKriging3D.py       # GPU加速3D克里金实现
│   └── ...
├── PINN/                      # 原有PINN模块
│   ├── tools.py               # PINN工具集
│   └── ...
├── ComposeProject/            # 新增耦合模块
│   ├── ComposeTools.py        # 耦合工具核心模块
│   ├── main.py               # 主程序入口
│   ├── docs/                 # 技术文档
│   ├── tests/                # 测试套件
│   └── environment.yml       # 环境配置
└── README.md                 # 本文件
```

## 快速开始

### 环境配置

```bash
# 创建conda环境
conda env create -f ComposeProject/environment.yml
conda activate gpu-kriging-pinn-coupling

# 设置pre-commit钩子
cd ComposeProject
pre-commit install
```

## GPU Kriging × PINN 耦合快速开始

### 1. 环境检查和通用工具演示

```bash
cd ComposeProject
python main.py --mode common --num_samples 300 --save_plots
```

### 2. 方案1演示: PINN → 残差Kriging → 加权融合

```bash
python main.py --mode mode1 \
    --num_samples 300 \
    --fusion_weight 0.6 \
    --pinn_epochs 2000 \
    --save_plots --show_plots
```

### 3. 方案2演示: Kriging ROI样本扩充 → PINN重训练

```bash
python main.py --mode mode2 \
    --num_samples 300 \
    --roi_strategy high_density \
    --augment_factor 2.5 \
    --pinn_epochs 2000 \
    --save_plots --show_plots
```

### 4. 测试和代码质量

```bash
# 运行完整测试套件
pytest tests/test_compose.py -v --cov=ComposeTools --cov-report=html

# 代码格式化
black ComposeTools.py main.py
isort ComposeTools.py main.py
```

## 耦合方案对比

| 特性 | 方案1 (残差融合) | 方案2 (样本扩充) |
|------|------------------|------------------|
| **核心思路** | PINN+Kriging残差修正 | Kriging增广→PINN重训练 |
| **计算复杂度** | 中等 | 较高 |
| **适用场景** | 快速修正、实时应用 | 离线训练、高精度需求 |

## 许可证

本项目采用 MIT 许可证

---

⭐ 如果本项目对您有帮助，请给我们一个Star！ 