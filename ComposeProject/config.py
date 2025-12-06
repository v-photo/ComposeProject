"""
系统配置文件 (字典格式)
"""
import numpy as np
from typing import Dict, Any

# --- 默认配置 ---
DEFAULT_CONFIG = {
    "experiment": {
        "name": "default_adaptive_pinn",
    },
    # 自适应实验配置（新增）
    "adaptive_experiment": {
        # 训练循环
        "total_epochs": 1000,
        "adaptive_cycle_epochs": 200,       # 每个自适应周期的最大训练轮数
        "detect_every": 100,                # 早停检测间隔
        "num_residual_scout_points": 5000,  # 残差侦察点数
        "exploration": {                    # 探索率递减策略
            "initial": 0.2,
            "final": 0.05,
            "decay": 0.02
        },
        # 开关
        "enable_kriging": True,
        "enable_data_injection": False,
        "enable_rapid_improvement_early_stop": True,
        # 数据拆分
        "split_ratios": [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # 主集 + 多储备
        "test_set_size": 300,
        # 基线对比
        "enable_baseline": True,
        # 输出文件名后缀
        "file_suffix": "full_adaptive"
    },
    "data": {
        "file_path": "../PINN/DATA.xlsx",
        "sheet_name_template": "avg_1_z",
        "use_columns": "B:EG",
        "z_size": 72,
        "y_size": 136,
        "space_dims": [20.0, 10.0, 10.0], # 物理空间维度 (x,y,z)
        "num_samples": 50, #修改取样点数
        "downsample_factor": 1, # 1表示不降采样
    },
    # 采样配置（新增）
    "sampling": {
        # 当要对比PINN和Kriging时，选kriging_style
        "strategy": "positive_only",  # 可选: "kriging_style", "positive_only", "uniform", "high_dose"
        # Kriging风格采样参数
        "kriging_style": {
            "box_origin": [5, 5, 5],      # 采样区域起点 [x, y, z] (网格索引)
            "box_extent": [126, 126, 62], # 采样区域延伸长度 [x_len, y_len, z_len]
            "step_sizes": [5],            # 采样步长列表
            "source_positions": [],       # 源点位置列表，用于排除
            "source_exclusion_radius": 30.0,  # 源点排除半径
        },
        # 随机采样参数（当strategy不是kriging_style时使用）
        "random_sampling": {
            "num_samples": 300,
        }
    },
    "pinn": {
        "model_params": {
            "network_layers": [3, 64, 64, 64, 1],   # 网络结构
            "learning_rate": 1e-3,                  # 学习率
            "loss_ratio": 10.0,                     # 数据/物理损失权重比
            "num_collocation_points": 4096,         # 配点数量
        },
        "training_params": {
            "total_epochs": 5000,                   # 兼容字段：总步数预算
            "cycle_epochs": 5000,                   # 首轮/单轮训练步数（推荐使用）
            "detect_every": 100,                    # 日志与早停检测间隔
            "adaptive_cycle_epochs": 2000,          # 第二阶段（自适应）训练步数
            "detection_threshold": 0.2              # 早停阈值（相对改善比例）
        },
        "physical_params": {
            'rho': 1.2,                             # 物理参数示例
            'mu': 1e-3
        }
    },
    "kriging": {
        "variogram_model": "exponential",           # 变异函数模型
        "nlags": 8,                                # 距离分组数
        "block_size": 10000,                       # 预测分块大小
        "exploration_ratio": 0.2,                  # Compose 自适应探索率
        "total_candidates": 50000,                 # Compose 候选点数量
        "enable_plotting": False,                  # 训练时是否绘图
        "weight": False,                           # 是否加权
        "uk": False,                               # Universal Kriging 开关
        "style": "gpu_b",                          # 预测风格
        "multi_process": False,                    # 预测是否多进程
        "print_time": False,                       # 是否打印耗时
        "torch_ac": False                          # torch 加速选项
    },
    "selection": {
        "min_points_for_kriging": 100,             # 自动选择时最少点数
        "uniformity_cv_threshold": 0.6,            # 最近邻CV阈值
    },
    "system": {
        "use_gpu": True,                           # 是否使用GPU
        "random_seed": 42,                         # 随机种子
        "verbose": True,                           # 详细日志
        "save_results": True,                      # 是否保存结果
        "checkpoint_path": "./models/pinn_checkpoint", # 检查点路径前缀
        "results_dir": "results",                  # 结果输出目录
        "method": "auto",                          # 默认方法，可被CLI覆盖
        "enable_compose_adaptive": False,          # Compose 残差引导配点
        "enable_pinn_adaptive": False,             # PINN 第二阶段随机加密
        "enable_data_injection": False             # 数据注入开关（需自行调用）
    }
}

# --- 预设 ---
PRESETS = {
    "default": DEFAULT_CONFIG,
    "quick_test": {
        **DEFAULT_CONFIG,
        "experiment": {"name": "quick_test"},
        "sampling": {
            "strategy": "kriging_style",
            "kriging_style": {
                "box_origin": [5, 5, 5],
                "box_extent": [60, 60, 30],  # 较小区域用于快速测试
                "step_sizes": [10],          # 较大步长
                "source_positions": [],
                "source_exclusion_radius": 30.0,
            },
            "random_sampling": {"num_samples": 50}
        },
        "pinn": {
            **DEFAULT_CONFIG["pinn"],
            "model_params": {
                **DEFAULT_CONFIG["pinn"]["model_params"],
                "num_collocation_points": 1024,
            },
            "training_params": {
                "total_epochs": 1000,
                "detect_every": 200,
                "adaptive_cycle_epochs": 500,
            }
        },
        "data": {
            **DEFAULT_CONFIG["data"],
            "num_samples": 50,
        },
        "system": {
            **DEFAULT_CONFIG["system"],
            "method": "auto"
        }
    },
    "kriging_only": {
        **DEFAULT_CONFIG,
        "experiment": {"name": "kriging_only_test"},
        "sampling": {
            "strategy": "kriging_style",
            "kriging_style": {
                "box_origin": [5, 5, 5],
                "box_extent": [126, 126, 62],
                "step_sizes": [5],
                "source_positions": [],
                "source_exclusion_radius": 30.0,
            },
            "random_sampling": {"num_samples": 300}
        },
        "selection": {
            "min_points_for_kriging": 1,
            "uniformity_cv_threshold": 999.0, # 强制选择Kriging
        },
        "system": {
            **DEFAULT_CONFIG["system"],
            "method": "kriging"
        }
    },
    "pinn_only": {
        **DEFAULT_CONFIG,
        "experiment": {"name": "pinn_only_test"},
        "sampling": {
            "strategy": "kriging_style",
            "kriging_style": {
                "box_origin": [5, 5, 5],
                "box_extent": [126, 126, 62],
                "step_sizes": [5],
                "source_positions": [],
                "source_exclusion_radius": 30.0,
            },
            "random_sampling": {"num_samples": 300}
        },
        "selection": {
            "min_points_for_kriging": 99999, # 强制选择PINN
            "uniformity_cv_threshold": 0.0,
        },
        "system": {
            **DEFAULT_CONFIG["system"],
            "method": "pinn"
        }
    },
    # 新增：使用随机采样的预设（保持向后兼容）
    "random_sampling": {
        **DEFAULT_CONFIG,
        "experiment": {"name": "random_sampling_test"},
        "sampling": {
            "strategy": "positive_only",
            "kriging_style": DEFAULT_CONFIG.get("sampling", {}).get("kriging_style", {}),
            "random_sampling": {"num_samples": 300}
        },
        "system": {
            **DEFAULT_CONFIG["system"],
            "method": "auto"
        }
    }
}

def load_config_dict(preset_name: str = "default") -> Dict[str, Any]:
    """
    根据预设名称加载配置字典。

    Args:
        preset_name (str): 预设名称。

    Returns:
        Dict[str, Any]: 配置字典。
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {list(PRESETS.keys())}")
    
    print(f"  ✅ Configuration loaded for preset: '{preset_name}'")
    return PRESETS[preset_name]
