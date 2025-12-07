import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import AdaptiveDataLoader
from src.models.pinn import PINNModel
from src.training.samplers import GpuKrigingSurrogate, AdaptiveSampler
from src.utils.environment import validate_compose_environment


def _compute_exploration_ratio(cycle_number: int, initial: float, final: float, decay: float) -> float:
    """按周期递减的探索率计算。"""
    return max(final, initial - (cycle_number - 1) * decay)


def _format_float(value: float, precision: int = 4) -> str:
    """安全格式化浮点数，便于表格展示。"""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _write_comparison_markdown(
    md_path: Path,
    exp_name: str,
    suffix: str,
    adaptive_stats: Dict[str, Any],
    baseline_stats: Dict[str, Any],
):
    """将耗时与精度的对比结果落盘为 Markdown。"""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        ("自适应PINN", adaptive_stats),
        ("基线PINN", baseline_stats),
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# PINN 对比汇总（{exp_name}）\n\n")
        f.write(f"- 生成时间：{timestamp}\n")
        f.write(f"- 配置后缀：{suffix}\n\n")
        f.write("| 模型 | 最终MRE | 最佳MRE | 训练轮数 | 耗时(秒) | 耗时(分钟) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for name, stats in rows:
            if not stats:
                final_mre = best_mre = epochs = time_sec = time_min = "N/A"
            else:
                final_mre = _format_float(stats.get("final_mre"), 6)
                best_mre = _format_float(stats.get("best_mre"), 6)
                epochs_val = stats.get("epochs")
                epochs = epochs_val if epochs_val is not None else "N/A"
                time_sec_val = stats.get("time_seconds")
                time_sec = _format_float(time_sec_val, 2)
                time_min = _format_float(
                    time_sec_val / 60 if isinstance(time_sec_val, (int, float)) else None,
                    2,
                )
            f.write(f"| {name} | {final_mre} | {best_mre} | {epochs} | {time_sec} | {time_min} |\n")

        f.write("\n> 说明：耗时统计覆盖模型初始化后的主要训练过程。\n")


def run_adaptive_experiment(config: Dict[str, Any]):
    """
    复刻 V1 的自适应循环：PINN 训练 -> 数据注入 -> Kriging 残差侦察 + 自适应采样 -> 循环。
    结束后用自适应实际训练点训练基线 PINN，对比并输出图。
    """
    print_compose_banner = None
    try:
        from src.utils.display import print_compose_banner as _banner
        print_compose_banner = _banner
    except Exception:
        pass

    if print_compose_banner:
        print_compose_banner()

    dep_status = validate_compose_environment()
    print("\n--- 📦 依赖状态检查 ---")
    for dep, status in dep_status.items():
        print(f"  - {dep}: {'✅ 可用' if status else '❌ 不可用'}")

    # 读取配置
    exp_cfg = config.get("adaptive_experiment", {})
    data_cfg = config.get("data", {})
    pinn_cfg = config.get("pinn", {})
    kriging_cfg = config.get("kriging", {})
    system_cfg = config.get("system", {})

    total_epochs = exp_cfg.get("total_epochs", 1000)
    cycle_epochs = exp_cfg.get("adaptive_cycle_epochs", 200)
    detect_every = exp_cfg.get("detect_every", 100)
    scout_points_num = exp_cfg.get("num_residual_scout_points", 5000)
    exploration_cfg = exp_cfg.get("exploration", {})
    enable_kriging = exp_cfg.get("enable_kriging", True)
    enable_data_injection = exp_cfg.get("enable_data_injection", False)
    enable_ries = exp_cfg.get("enable_rapid_improvement_early_stop", True)
    split_ratios = exp_cfg.get("split_ratios", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    test_set_size = exp_cfg.get("test_set_size", 300)
    enable_baseline = exp_cfg.get("enable_baseline", True)
    file_suffix = exp_cfg.get("file_suffix")  # 如未设定，将根据开关动态生成

    np.random.seed(system_cfg.get("random_seed", 42))

    # 数据加载与拆分
    data_loader = AdaptiveDataLoader(
        data_path=data_cfg.get("file_path"),
        space_dims=np.array(data_cfg.get("space_dims", [20.0, 10.0, 10.0])),
        num_samples=data_cfg.get("num_samples", 300),
    )
    main_train, reserve_pools, test_set, dose_data = data_loader.get_training_data(
        split_ratios=split_ratios,
        test_set_size=test_set_size,
    )

    # 预测网格
    from src.data.loader import create_prediction_grid
    prediction_points = create_prediction_grid(
        dose_data=dose_data,
        downsample_factor=data_cfg.get("downsample_factor", 1),
    )

    # 初始化 PINN
    model_params = pinn_cfg.get("model_params", {})
    training_params = pinn_cfg.get("training_params", {})
    num_collocation = model_params.get("num_collocation_points", 4096)
    detect_every = detect_every or training_params.get("detect_every", 500)

    pinn = PINNModel(
        dose_data=dose_data,
        training_data=main_train,
        test_data=test_set,
        **model_params,
    )

    # 初始 collocation
    current_collocation_points = np.random.uniform(
        low=dose_data["world_min"],
        high=dose_data["world_max"],
        size=(num_collocation, 3),
    )

    sampler = None
    surrogate = None
    if enable_kriging:
        sampler = AdaptiveSampler(
            domain_bounds=np.vstack([dose_data["world_min"], dose_data["world_max"]]),
            total_candidates=kriging_cfg.get("total_candidates", 50000),
        )
        surrogate = GpuKrigingSurrogate(
            variogram_model=kriging_cfg.get("variogram_model", "exponential"),
            nlags=kriging_cfg.get("nlags", 8),
            block_size=kriging_cfg.get("block_size", 10000),
        )

    important_events: List[Tuple[int, str, str]] = []

    total_epochs_trained = 0
    cycle_counter = 0
    history_epochs = []
    history_mre = []

    adaptive_start_time = time.time()
    while total_epochs_trained < total_epochs:
        remaining_total = total_epochs - total_epochs_trained
        cycle_max = min(cycle_epochs, remaining_total)

        print(f"\n--- 主循环周期: 目标训练 {total_epochs_trained} -> {total_epochs_trained + cycle_max} ---")

        epochs_before = pinn.model.train_state.step or 0
        cycle_result = pinn.run_training_cycle(
            max_epochs=cycle_max,
            detect_every=detect_every,
            detection_threshold=training_params.get("detection_threshold", 0.1),
            collocation_points=current_collocation_points,
            checkpoint_path_prefix=system_cfg.get("checkpoint_path", "./models/pinn_checkpoint"),
        )
        epochs_after = pinn.model.train_state.step or 0
        epochs_this_cycle = epochs_after - epochs_before
        total_epochs_trained += epochs_this_cycle
        cycle_counter += 1

        # 记录阶段事件 + 周期内早停/回退事件
        if pinn.epoch_history:
            important_events.append((pinn.epoch_history[-1], "phase_transition", f"周期{cycle_counter}完成"))
        if cycle_result and cycle_result.get("events"):
            for e_step, e_type in cycle_result["events"]:
                desc = "早停" if e_type == "early_stop" else "回退" if e_type == "rollback" else "训练事件"
                important_events.append((e_step, e_type, desc))

        # 记录训练曲线
        history_epochs = pinn.epoch_history
        history_mre = pinn.mre_history

        if total_epochs_trained >= total_epochs:
            print("INFO: 总训练轮数已达目标，结束。")
            break

        # 数据注入
        if enable_data_injection:
            if reserve_pools:
                data_injection_epoch = pinn.model.train_state.step or 0
                data_to_inject = reserve_pools.pop(0)
                pinn.inject_new_data(data_to_inject)
                important_events.append(
                    (data_injection_epoch, "data_injection", f"周期{cycle_counter}数据注入(+{len(data_to_inject)}点)")
                )
            else:
                print("WARNING: 数据注入已启用但无储备数据。")

        # Kriging 重采样
        if enable_kriging and sampler and surrogate:
            print("\nPHASE: Kriging 残差侦察与自适应采样")
            scout_points = np.random.uniform(
                low=dose_data["world_min"],
                high=dose_data["world_max"],
                size=(scout_points_num, 3),
            )
            true_residuals = pinn.compute_pde_residual(scout_points)
            surrogate.fit(scout_points, true_residuals)

            exploration_ratio = _compute_exploration_ratio(
                cycle_number=cycle_counter,
                initial=exploration_cfg.get("initial", 0.2),
                final=exploration_cfg.get("final", 0.05),
                decay=exploration_cfg.get("decay", 0.02),
            )
            current_collocation_points = sampler.generate_new_collocation_points(
                surrogate_model=surrogate,
                num_points_to_sample=num_collocation,
                exploration_ratio=exploration_ratio,
            )
            kriging_epoch = pinn.model.train_state.step or 0
            important_events.append(
                (kriging_epoch, "kriging_resampling", f"周期{cycle_counter}克里金采样(探索率={exploration_ratio:.2f})")
            )
        else:
            print("PHASE: Kriging 自适应采样已禁用，保持现有配置点。")

    adaptive_time = time.time() - adaptive_start_time
    print(f"\n--- ✅ 自适应训练完成，耗时 {adaptive_time/60:.2f} 分 ---")

    adaptive_summary = {
        "final_mre": history_mre[-1] if history_mre else None,
        "best_mre": float(np.min(history_mre)) if history_mre else None,
        "epochs": total_epochs_trained,
        "time_seconds": adaptive_time,
    }

    # 基线对比
    baseline_history = None
    baseline_summary = None
    baseline_time = None
    if enable_baseline:
        print("\n--- 🚀 训练基线 PINN (固定采样) ---")
        baseline_start_time = time.time()
        adaptive_training_points = pinn.data.bcs[0].points
        adaptive_training_values = np.exp(pinn.data.bcs[0].values.cpu().numpy())
        full_training_data = np.hstack([adaptive_training_points, adaptive_training_values])

        baseline = PINNModel(
            dose_data=dose_data,
            training_data=full_training_data,
            test_data=test_set,
            **model_params,
        )
        baseline_collocation = np.random.uniform(
            low=dose_data["world_min"],
            high=dose_data["world_max"],
            size=(num_collocation, 3),
        )
        if baseline.model.train_state.X_train is None:
            baseline.model.train(iterations=0)
        num_bc = baseline.data.bcs[0].points.shape[0]
        start_idx = num_bc
        end_idx = len(baseline.model.train_state.X_train) - len(baseline.data.anchors)
        baseline.model.train_state.X_train[start_idx:end_idx] = baseline_collocation
        baseline.model.train(iterations=total_epochs, display_every=detect_every)

        baseline_history = {
            "Baseline PINN": {
                "epochs": baseline.epoch_history,
                "metrics": baseline.mre_history,
            }
        }
        baseline_time = time.time() - baseline_start_time
        baseline_summary = {
            "final_mre": baseline.mre_history[-1] if baseline.mre_history else None,
            "best_mre": float(np.min(baseline.mre_history)) if baseline.mre_history else None,
            "epochs": baseline.model.train_state.step or 0,
            "time_seconds": baseline_time,
        }
        print(f"--- ✅ 基线 PINN 训练完成，耗时 {baseline_time/60:.2f} 分 ---")

    # 汇总历史并绘图
    history = {
        "Adaptive PINN": {
            "epochs": history_epochs,
            "metrics": history_mre,
            "events": important_events,
        }
    }
    if baseline_history:
        history.update(baseline_history)

    results_dir = Path(system_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    exp_name = config.get("experiment", {}).get("name", "adaptive_experiment")
    events = important_events

    # 动态文件后缀与描述（与 V1 保持一致）
    if file_suffix:
        suffix = file_suffix
    else:
        if enable_kriging and enable_data_injection:
            suffix = "full_adaptive"
        elif enable_kriging and not enable_data_injection:
            suffix = "kriging_only"
        elif (not enable_kriging) and enable_data_injection:
            suffix = "data_injection_only"
        else:
            suffix = "periodic_restart"

    # 使用与 V1 风格一致的绘图，输出 png/pdf
    png_path = results_dir / f"mre_comparison_{suffix}.png"
    pdf_path = results_dir / f"mre_comparison_{suffix}.pdf"
    _plot_v1_style(
        adaptive_history=history.get("Adaptive PINN"),
        baseline_history=history.get("Baseline PINN"),
        events=events,
        suffix=suffix,
        save_png=png_path,
        save_pdf=pdf_path,
    )

    md_path = results_dir / f"pinn_comparison_{suffix}.md"
    _write_comparison_markdown(
        md_path=md_path,
        exp_name=exp_name,
        suffix=suffix,
        adaptive_stats=adaptive_summary,
        baseline_stats=baseline_summary,
    )

    np.savez(
        results_dir / f"training_history_{exp_name}.npz",
        epochs=np.array(history_epochs),
        metrics=np.array(history_mre),
        events=np.array(events, dtype=object),
    )

    print(f"\n🎉 实验完成。结果已保存至 {results_dir}")


def _plot_v1_style(
    adaptive_history: Dict[str, Any],
    baseline_history: Dict[str, Any],
    events: List[Tuple[int, str, str]],
    suffix: str,
    save_png: Path,
    save_pdf: Path,
):
    """复刻 V1 的简单绘图风格，便于对比。"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 自适应曲线
    if adaptive_history:
        ax.plot(
            adaptive_history.get("epochs", []),
            adaptive_history.get("metrics", []),
            label="自适应PINN",
            linewidth=2,
            alpha=0.8,
            color="blue",
        )

    # 基线曲线
    if baseline_history:
        ax.plot(
            baseline_history.get("epochs", []),
            baseline_history.get("metrics", []),
            label="原始PINN (固定采样)",
            linewidth=2,
            alpha=0.8,
            color="red",
        )

    # 事件标注（仅 data_injection/kriging_resampling，颜色与 V1 对齐；错峰高度避免重叠）
    if events:
        event_styles = {
            "data_injection": {"color": "green", "linestyle": "--", "alpha": 0.7},
            "kriging_resampling": {"color": "orange", "linestyle": "-.", "alpha": 0.7},
        }
        for i, (epoch, event_type, desc) in enumerate(events):
            style = event_styles.get(event_type)
            if not style:
                continue
            ax.axvline(x=epoch, **style, linewidth=2)
            y_min, y_max = ax.get_ylim()
            # 预设多个高度层，交替使用，避免重叠
            levels = [0.82, 0.68, 0.54, 0.4, 0.26]
            y_pos = y_max * levels[i % len(levels)]
            ax.annotate(
                f"{desc}\n(Epoch {epoch})",
                xy=(epoch, y_pos),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=style["color"], alpha=0.3),
                fontsize=9,
                ha="left",
            )

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="green", linestyle="--", label="数据注入"),
            Line2D([0], [0], color="orange", linestyle="-.", label="克里金重采样"),
        ]
        second_legend = ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=10,
            title="重要事件",
            title_fontsize=11,
        )
        ax.add_artist(second_legend)

    ax.set_xlabel("训练轮数 (Epochs)", fontsize=12)
    ax.set_ylabel("平均相对误差 (MRE)", fontsize=12)
    ax.set_title(f"PINN 训练过程对比 ({suffix})", fontsize=14, fontweight="bold")
    ax.legend(loc="center right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    save_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close(fig)
