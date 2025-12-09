"""
在 auto 方法下，训练模型并评估全场预测准确率，生成真实值 vs 预测值的三维剂量对比图。

运行示例：
    python Test/field_3d_comparison.py --preset default

说明：
- 训练集与预测网格由 config.py 中的预设决定（数据文件、采样策略等）。
- 自动工作流会根据采样点分布选择 Kriging 或 PINN。
- 输出：打印指标，并在 results 目录下保存对比图和指标 JSON。
"""

import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

# 确保可以导入项目内模块（包含上级根目录以找到 Kriging）
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import load_config_dict
from src.data.loader import (
    load_3d_data_from_sheets,
    process_grid_to_dose_data,
    sample_training_points,
    sample_kriging_style,
    create_prediction_grid,
)
from src.workflows.auto_selection import AutoSelectionWorkflow
from src.analysis.metrics import MetricsCalculator

# 设置中文字体，避免乱码
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

def get_training_samples(dose_data, config):
    """与 main.py 保持一致的采样逻辑。"""
    sampling_cfg = config.get("sampling", {})
    strategy = sampling_cfg.get("strategy", "positive_only")

    if strategy == "kriging_style":
        kriging_cfg = sampling_cfg.get("kriging_style", {})
        train_points, train_values = sample_kriging_style(
            dose_data,
            box_origin=kriging_cfg.get("box_origin", [5, 5, 5]),
            box_extent=kriging_cfg.get("box_extent", [90, 90, 90]),
            step_sizes=kriging_cfg.get("step_sizes", [5]),
            source_positions=kriging_cfg.get("source_positions", None),
            source_exclusion_radius=kriging_cfg.get("source_exclusion_radius", 30.0),
        )
    else:
        random_cfg = sampling_cfg.get("random_sampling", {})
        num_samples = random_cfg.get("num_samples", config.get("data", {}).get("num_samples", 300))
        train_points, train_values = sample_training_points(
            dose_data=dose_data,
            num_samples=num_samples,
            strategy=strategy if strategy != "kriging_style" else "positive_only",
        )
    return train_points, train_values


def plot_true_vs_pred(
    points,
    true_vals,
    pred_vals,
    save_path,
    title,
    mre_value,
    sample_size=5000,
    use_log_norm=True,
    clip_percentiles=(1, 99),
):
    """
    绘制真实 vs 预测三维散点图（随机下采样），并标注 MRE。
    支持对颜色做分位裁剪和对数归一，缓解低值占多数导致的对比不足。
    """
    n = len(points)
    if n == 0:
        raise ValueError("无预测点可绘图。")
    idx = np.random.choice(n, min(sample_size, n), replace=False)
    pts = points[idx]
    tvals = true_vals[idx]
    pvals = pred_vals[idx]

    combined = np.concatenate([tvals, pvals])
    p_low, p_high = clip_percentiles
    vmin = np.percentile(combined, p_low)
    vmax = np.percentile(combined, p_high)
    # 防止 vmin==vmax 或非正值导致 LogNorm 失败
    if vmin <= 0:
        vmin = max(vmin, 1e-8)
    if vmin >= vmax:
        vmin = combined.min()
        vmax = combined.max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if use_log_norm and vmin > 0 else mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=tvals, cmap="viridis", norm=norm, s=6)
    ax1.set_title("真实剂量")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    fig.colorbar(sc1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc2 = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pvals, cmap="viridis", norm=norm, s=6)
    ax2.set_title("预测剂量")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    fig.colorbar(sc2, ax=ax2, shrink=0.6)

    fig.suptitle(f"{title}\nMRE={mre_value:.4f}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ 图已保存: {save_path}")


def run(preset: str):
    # 1. 加载配置
    config = load_config_dict(preset)
    # 强制使用 auto 模式，保证走自动选择分支
    config["system"]["method"] = "auto"
    # 指定图片/指标输出路径到“三维辐射场对比图”
    custom_results_dir = PROJECT_DIR / "results" / "三维辐射场对比图"
    config["system"]["results_dir"] = str(custom_results_dir)

    # 2. 数据加载
    data_cfg = config.get("data", {})
    dose_grid = load_3d_data_from_sheets(
        file_path=data_cfg.get("file_path"),
        sheet_name_template=data_cfg.get("sheet_name_template"),
        use_cols=data_cfg.get("use_columns"),
        z_size=data_cfg.get("z_size"),
        y_size=data_cfg.get("y_size"),
    )
    dose_data = process_grid_to_dose_data(
        dose_grid=dose_grid,
        space_dims=data_cfg.get("space_dims"),
    )

    # 3. 采样训练集 & 预测网格
    train_points, train_values = get_training_samples(dose_data, config)
    prediction_points = create_prediction_grid(
        dose_data=dose_data,
        downsample_factor=data_cfg.get("downsample_factor", 1),
    )

    # 4. 自动工作流训练与预测
    workflow = AutoSelectionWorkflow(config)
    results = workflow.run(
        train_points=train_points,
        train_values=train_values,
        prediction_points=prediction_points,
        dose_data=dose_data,
    )
    predictions = results.get("predictions")
    method_used = results.get("method_used", "auto")
    if predictions is None:
        raise RuntimeError("自动工作流未返回预测结果。")

    # 5. 计算指标
    true_field = dose_data["dose_grid"].flatten()
    metrics = MetricsCalculator.compute_metrics(true_values=true_field, pred_values=predictions)
    mre_stats = MetricsCalculator.compute_relative_error_stats(true_values=true_field, pred_values=predictions)
    mre_value = metrics.get("MRE", float("nan"))

    # 6. 绘图
    results_dir = custom_results_dir
    exp_name = config.get("experiment", {}).get("name", preset)
    fig_path = results_dir / f"field_3d_comparison_{exp_name}.png"
    plot_true_vs_pred(
        points=prediction_points,
        true_vals=true_field,
        pred_vals=predictions,
        save_path=fig_path,
        title=f"Auto 方法 ({method_used}) 真实 vs 预测",
        mre_value=mre_value,
    )

    # 7. 保存指标
    summary = {
        "preset": preset,
        "method_used": method_used,
        "metrics": metrics,
        "relative_error_stats": mre_stats,
        "train_points": len(train_points),
        "prediction_points": len(prediction_points),
    }
    metrics_path = results_dir / f"field_3d_metrics_{exp_name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ 指标已保存: {metrics_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Auto 方法三维剂量对比测试")
    parser.add_argument("--preset", type=str, default="default", help="config.py 中的预设名称")
    args = parser.parse_args()
    run(args.preset)


if __name__ == "__main__":
    main()

