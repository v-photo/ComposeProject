下面是一份可交给其他 AI 的协作指令，覆盖目标与关键参考：

目标
- 在保留重构版框架（耦合项目V2）的基础上，移植/对齐历史项目 `ComposeProject/example2.py` 中的“自适应 PINN + 数据注入 + Kriging 重采样 + 周期控制 + 事件标注”机制。
- 保留现有四种模式（auto/kriging/pinn/compose）；耦合模式在此基础上继续存在。
- 输出训练曲线与事件标注要与历史对比脚本效果一致（如 `mre_comparison_data_injection_only.png`）。

关键参考（历史项目）
- 文件：`耦合项目/ComposeProject/example2.py`（完整的自适应循环与对比图生成逻辑）。
  - 全局开关：`ENABLE_KRIGING`、`ENABLE_DATA_INJECTION`、`ENABLE_RAPID_IMPROVEMENT_EARLY_STOP`，探索率递减策略。
  - 训练循环：总轮数 `TOTAL_EPOCHS`，每周期 `ADAPTIVE_CYCLE_EPOCHS`，侦察点数 `NUM_RESIDUAL_SCOUT_POINTS`。
  - 阶段：
    1) PINN 训练一周期（带早停/回退）；记录 MRE。
    2) 周期干预：
       - 若启用数据注入：从 reserve pool 注入，记录事件 `data_injection`。
       - 若启用 Kriging：残差侦察 → Kriging 代理 → 自适应采点 → 记录事件 `kriging_resampling`。
    3) 循环至总轮数。
  - 基线对比：再训练一个基线 PINN，使用自适应 PINN 实际使用过的训练数据，生成对比图。
  - 事件标注：在图中对数据注入、Kriging 重采样标注垂直线。
  - 输出：`ComposeProject/results/mre_comparison_*.png/pdf` 动态命名。

当前重构版现状（耦合项目V2）
- 已有 `main.py` 支持 auto/kriging/pinn/compose，事件标注（early_stop/rollback）和图保存 `results/training_history_<exp>.png`。
- 配置在 `config.py`，开关 `enable_pinn_adaptive`、`enable_compose_adaptive`，早停阈值 `detection_threshold`。
- 早停/回退事件已记录，但没有历史版的“周期性数据注入 + Kriging 重采样 + 基线对比”流程。

待实现/移植要点
1) 引入“自适应实验脚本”或工作流，复刻 `example2.py` 的主循环：
   - 参数：TOTAL_EPOCHS、ADAPTIVE_CYCLE_EPOCHS、NUM_RESIDUAL_SCOUT_POINTS、探索率递减策略（INITIAL/FINAL/DECAY）。
   - 开关：ENABLE_KRIGING、ENABLE_DATA_INJECTION、ENABLE_RAPID_IMPROVEMENT_EARLY_STOP。
   - 数据加载与拆分（主训练/储备池/独立测试集）参考 `DummyDataLoader.get_training_data` 及测试集生成。
   - 训练步骤：pinn.run_training_cycle（带早停/回退）→ 注入/重采样 → 下一周期。
   - 事件：数据注入、Kriging 重采样、早停/回退（已有）、阶段完成，写入重要事件并用于图标注。
   - 图输出：动态文件名 `mre_comparison_{full_adaptive|kriging_only|data_injection_only|periodic_restart}.png/pdf`，标注事件，双模型对比（自适应 vs 基线）。

2) 兼容现有框架：
   - 可新增一个脚本或命令行模式（例如 `python main.py --mode adaptive_experiment`），调用上述循环，不破坏现有四模式。
   - 或单独放在 `scripts/adaptive_experiment.py`，但要重用 V2 的 `PINNModel`、KrigingAdapter 或 GPUKrigingSurrogate（如需），以及 plotting 事件标注函数。

3) 配置映射：
   - 将 `example2.py` 中的硬编码常量映射到 `config.py` 的新字段（adaptive_experiment 节），含开关、周期、侦察点数、探索率策略、分割比例、测试集大小、基线对比开关。
   - CLI 可允许覆盖关键参数。

4) 数据注入：
   - 重用 V2 的 `PINNModel.inject_new_data`（已存在）或按历史逻辑追加到训练集并重新编译。
   - 储备池从数据加载阶段拆分得来。

5) Kriging 重采样：
   - 残差侦察 → `GpuKrigingSurrogate` 或现 `KrigingAdapter`（需支持残差预测） → `AdaptiveSampler`（历史逻辑 Hard-Case Mining + Exploration） → 更新 collocation_points。

6) 基线对比：
   - 训练第二个 PINN 基线模型，使用自适应模型实际训练点，固定配置点，输出与自适应的对比 MRE。

7) 事件标注与输出：
   - 事件类型：data_injection、kriging_resampling、early_stop、rollback、phase_transition。
   - 图表样式参考历史脚本，保存 PNG/PDF 到 `ComposeProject/results/`。

交付预期
- 新增或扩展一个可运行入口，执行完整自适应对比实验，生成 `mre_comparison_*.png/pdf`。
- 配置可在 `config.py` 中集中管理，开关可通过 CLI 覆盖。
- 保留当前模式（auto/kriging/pinn/compose）不破坏。

可参考文件
- 历史：`耦合项目/ComposeProject/example2.py`（主循环、自适应逻辑、图生成）。
- 重构：`耦合项目V2/ComposeProject/main.py`、`src/models/pinn.py`、`src/training/samplers.py`、`src/analysis/plotting.py`、`config.py`。