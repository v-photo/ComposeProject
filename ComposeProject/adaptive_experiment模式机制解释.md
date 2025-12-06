# 仅适用于 `--method adaptive_experiment`

本说明专门描述 adaptive_experiment 模式下的各机制逻辑。

## 总览
- 多周期循环：总训练轮数按周期切分（默认 total_epochs=1000，周期长 adaptive_cycle_epochs=200）。
- 每周期流程：PINN 训练（带早停/回退）→ 可选数据注入 → 可选 Kriging 残差侦察 + 自适应采样 → 进入下一周期。
- 周期结束事件会记录：phase_transition、early_stop、rollback、data_injection、kriging_resampling。
- 结束后：用自适应实际训练过的点再训一个固定采样基线 PINN，绘制对比图 `mre_comparison_<suffix>.png/pdf`。

## 训练循环
1) 周期上限：`adaptive_cycle_epochs`（不超过剩余 total_epochs）。
2) PINN 训练：`run_training_cycle` 分段检查。
   - 每个 detect_every 块后，比较当前 MRE 与周期最佳 MRE。
   - 若 MRE 变差：回滚到周期最佳检查点，标记 `stagnation_detected`，提前结束本周期。
   - 若改善超过阈值（`detection_threshold`，默认 0.1，相对改善）：标记早停，提前结束本周期。
   - 周期结束后始终恢复最佳检查点并清理临时文件。
3) 事件记录：
   - `phase_transition`：周期结束。
   - `early_stop`：达到改善阈值的提前结束。
   - `rollback`：因性能变差回退到最佳模型的事件。

## 数据注入（可选）
- 开关：`adaptive_experiment.enable_data_injection`（默认 False）。
- 数据来源：`AdaptiveDataLoader` 按 `split_ratios` 预分的储备池，每周期取一池注入（若耗尽则跳过并警告）。
- 调用：`PINNModel.inject_new_data`，事件标记 `data_injection`。

## Kriging 残差侦察与自适应采样（可选）
- 开关：`adaptive_experiment.enable_kriging`（默认 True）。
- 侦察点：`num_residual_scout_points`（默认 5000），在物理域内均匀采样。
- 残差代理：`GpuKrigingSurrogate.fit(scout_points, true_residuals)`。
- 探索率递减：`exploration_ratio = max(final, initial - (cycle-1)*decay)`，参数来自 `exploration.initial/final/decay`（默认 0.2 → 0.05，步长 0.02）。
- 采样器：`AdaptiveSampler.generate_new_collocation_points`，按 Hard-case (exploitation) + 随机 (exploration) 生成等量新配点，替换下一周期的 collocation_points。
- 事件标记：`kriging_resampling`，注记探索率与周期号。

## 基线对比
- 开关：`adaptive_experiment.enable_baseline`（默认 True）。
- 训练数据：使用自适应模型实际训练过的全部点（含注入）。
- 配点：一次性随机固定。
- 训练轮数：与 total_epochs 对齐。
- 对比图：自适应 vs 基线，事件标记为数据注入/克里金重采样（早停/回退事件不单独上图，保持简洁）。

## 输出与命名
- 动态后缀（未显式指定 file_suffix 时自动判定）：
  - data_injection + kriging → `full_adaptive`
  - kriging only → `kriging_only`
  - data_injection only → `data_injection_only`
  - neither → `periodic_restart`
- 图：`results/mre_comparison_<suffix>.png`、`results/mre_comparison_<suffix>.pdf`
- 历史：`results/training_history_<exp>.npz`（含 epochs/metrics/events）

## 默认配置（与历史 V1 对齐）
```python
adaptive_experiment = {
  "total_epochs": 1000,
  "adaptive_cycle_epochs": 200,
  "detect_every": 100,
  "num_residual_scout_points": 5000,
  "exploration": {"initial": 0.2, "final": 0.05, "decay": 0.02},
  "enable_kriging": True,
  "enable_data_injection": False,
  "enable_rapid_improvement_early_stop": True,
  "split_ratios": [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
  "test_set_size": 300,
  "enable_baseline": True
}
```

