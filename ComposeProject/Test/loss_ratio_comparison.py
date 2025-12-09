"""
在不同 loss_ratio 下，对比自适应 PINN 与基线 PINN 的训练曲线并输出一张汇总图。

运行方式（示例）：
    python Test/loss_ratio_comparison.py

参数可在脚本顶部修改：
- PRESET: 使用的配置预设（默认 adaptive_experiment_config1，便于快速运行且不开 Kriging）
- LOSS_RATIOS: 要测试的 loss_ratio 列表
- RESULTS_DIR: 输出目录
"""

import copy
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 确保可以导入项目内模块
import sys
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import load_config_dict
from src.workflows.adaptive_experiment import run_adaptive_experiment


# ===== 可配置区域 =====
PRESET = "adaptive_experiment_config1"
LOSS_RATIOS = [0.1, 1.0, 5.0, 10.0]  # 可自行调整
# 结果输出目录（按照需求保存到“不同权重比例结果图”）
RESULTS_DIR = PROJECT_DIR / "results" / "不同权重比例结果图"


def run_one(loss_ratio: float, preset: str):
    cfg = copy.deepcopy(load_config_dict(preset))
    exp_name = f"loss_ratio_{loss_ratio}"
    cfg["experiment"]["name"] = exp_name
    # 设置 loss_ratio
    cfg["pinn"]["model_params"]["loss_ratio"] = loss_ratio
    # 确保基线开启，方便对比
    cfg["adaptive_experiment"]["enable_baseline"] = True
    # 将结果目录定向到指定子目录，确保 npz/图文件落在同一位置
    cfg["system"]["results_dir"] = str(RESULTS_DIR)

    print(f"\n=== Running loss_ratio={loss_ratio} ===")
    run_adaptive_experiment(cfg)

    npz_path = RESULTS_DIR / f"training_history_{exp_name}.npz"
    data = np.load(npz_path, allow_pickle=True)
    adaptive_epochs = data.get("adaptive_epochs", [])
    adaptive_metrics = data.get("adaptive_metrics", [])
    baseline_epochs = data.get("baseline_epochs", [])
    baseline_metrics = data.get("baseline_metrics", [])
    return {
        "loss_ratio": loss_ratio,
        "adaptive_epochs": adaptive_epochs,
        "adaptive_metrics": adaptive_metrics,
        "baseline_epochs": baseline_epochs,
        "baseline_metrics": baseline_metrics,
    }
    # 按需求清理 npz，避免残留
    try:
        npz_path.unlink()
    except FileNotFoundError:
        pass


def plot_all(runs, save_path: Path):
    plt.figure(figsize=(10, 6))
    # 为每个 loss_ratio 生成独立颜色，基线/自适应同色不同线型
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    for i, run in enumerate(runs):
        lr = run["loss_ratio"]
        color = colors[i % len(colors)]
        if len(run["adaptive_epochs"]) > 0:
            plt.plot(run["adaptive_epochs"], run["adaptive_metrics"], label=f"Adaptive lr={lr}", linewidth=2, color=color, linestyle="-")
        if len(run["baseline_epochs"]) > 0:
            plt.plot(run["baseline_epochs"], run["baseline_metrics"], label=f"Baseline lr={lr}", linewidth=2, color=color, linestyle="--")

    plt.xlabel("Epochs")
    plt.ylabel("MRE")
    plt.yscale("log")
    plt.title("Adaptive vs Baseline under different loss_ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved summary plot to: {save_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for lr in LOSS_RATIOS:
        runs.append(run_one(lr, PRESET))
    plot_all(runs, RESULTS_DIR / "loss_ratio_comparison.png")


if __name__ == "__main__":
    main()

