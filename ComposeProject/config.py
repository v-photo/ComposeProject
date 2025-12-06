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
    "data": {
        "file_path": "../PINN/DATA.xlsx",
        "sheet_name_template": "avg_1_z",
        "use_columns": "B:EG",
        "z_size": 72,
        "y_size": 136,
        "space_dims": [20.0, 10.0, 10.0], # 物理空间维度 (x,y,z)
        "num_samples": 300,
        "downsample_factor": 1, # 1表示不降采样
    },
    # 采样配置（新增）
    "sampling": {
        "strategy": "kriging_style",  # 可选: "kriging_style", "positive_only", "uniform", "high_dose"
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
            "network_layers": [3, 64, 64, 64, 1],
            "learning_rate": 1e-3,
            "loss_ratio": 10.0,
            "num_collocation_points": 4096,
        },
        "training_params": {
            "total_epochs": 5000,
            "detect_every": 500,
            "adaptive_cycle_epochs": 2000,
        },
        "physical_params": {
            'rho': 1.2,
            'mu': 1e-3
        }
    },
    "kriging": {
        "variogram_model": "exponential",
        "nlags": 8,
        "block_size": 10000,
    },
    "selection": {
        "min_points_for_kriging": 100,
        "uniformity_cv_threshold": 0.6,
    },
    "system": {
        "use_gpu": True,
        "random_seed": 42,
        "verbose": True,
        "save_results": True,
        "checkpoint_path": "./models/pinn_checkpoint"
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
