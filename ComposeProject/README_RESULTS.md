# 结果文件保存与标注指南

## 输出目录与命名
- 目录：默认 `results/`（可在 `config.py -> system.results_dir` 调整）。
- 实验名：取自 `config.py -> experiment.name`（或预设覆盖）。
- 自适应后缀：`adaptive_experiment.file_suffix`；未设置时按开关动态生成：
  - 同时开 Kriging + 注入 → `full_adaptive`
  - 仅 Kriging → `kriging_only`
  - 仅注入 → `data_injection_only`
  - 都关闭 → `periodic_restart`

## 常见文件
- `predictions_<experiment>.npy`：Kriging/PINN/Auto 的预测结果。
- `training_history_<experiment>.npz`：包含 epochs、metrics、events。
- `training_history_<experiment>.png`：训练曲线，可包含事件标注（见下；adaptive_experiment 现在也会自动生成）。
- 自适应实验特有：
  - `mre_comparison_<suffix>.png/pdf`：自适应 vs 基线曲线（V1 风格）。
  - `pinn_comparison_<suffix>.md`：自适应/基线的 MRE、耗时、训练点数摘要。

## 事件标注规则
- 自适应对比图（`mre_comparison_<suffix>.png/pdf`，函数 `_plot_v1_style`）：
  - 仅标注：`data_injection`、`kriging_resampling`。
  - 其他事件（phase_transition/early_stop/rollback）只存事件列表，不画线。
- 训练历史图（`training_history_*.png`，函数 `plot_training_comparison`）：
  - 有专门样式：`data_injection`、`kriging_resampling`、`phase_transition`、`loss_ratio_update`。
  - 其他事件（如 `early_stop`、`rollback`）若传入，则以灰色“其他”样式展示。

## 生成逻辑简述
- 主流程 `main.py`：
  - Kriging：仅保存 `predictions_*.npy`。
  - PINN / Compose：若有 `history`，保存 `training_history_*.png/npz`，事件来自训练周期记录。
  - Auto：按自动决策结果保存。
- 自适应实验 `run_adaptive_experiment`：
  - 记录自适应/基线两套曲线与事件；
  - 按上述规则输出对比图、训练历史和 Markdown 摘要。

## 自定义（进阶）
- 使用 `plot_training_comparison(models_history, important_events, title, save_path)` 可手工生成带事件标注的训练图：
  - `models_history` 示例：`{"ModelA": {"epochs": [...], "metrics": [...]}}`
  - `important_events` 示例：`[(500, "data_injection", "周期1注入"), (800, "kriging_resampling", "周期1克里金")]`
  - `save_path` 为空则直接显示，非空则保存文件。

### 如何手动调用 `plot_training_comparison`
```python
from src.analysis.plotting import plot_training_comparison

# 1) 准备训练曲线（必须有 epochs 与 metrics）
history = {
    "Adaptive PINN": {
        "epochs": [0, 100, 200, 300],
        "metrics": [0.5, 0.2, 0.12, 0.1],
    },
    "Baseline PINN": {
        "epochs": [0, 100, 200, 300],
        "metrics": [0.5, 0.25, 0.18, 0.16],
    },
}

# 2) 可选事件标注（epoch, event_type, desc）
events = [
    (120, "data_injection", "周期1注入"),
    (180, "kriging_resampling", "周期1克里金"),
    (300, "phase_transition", "周期1结束"),
]

# 3) 调用（save_path 为空则显示，非空则保存）
plot_training_comparison(
    models_history=history,
    important_events=events,
    title="手动示例",
    save_path="results/manual_plot.png"
)
```
事件类型样式：`data_injection`（绿虚线）、`kriging_resampling`（橙点划线）、`phase_transition`（紫点线）、`loss_ratio_update`（红实线）；其他类型将以灰色“其他”显示。

