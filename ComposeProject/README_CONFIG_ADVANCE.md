# 高级配置与机制开关说明

## 可选机制开关与模式（示例结构）
```python
@dataclass
class SystemConfig:
    method: str = "auto"                 # auto / kriging / pinn / compose / adaptive_experiment
    enable_compose_adaptive: bool = False
    enable_pinn_adaptive: bool = False
    enable_data_injection: bool = False  # 仅 compose/pinn 使用；adaptive_experiment 单独配置
```

### 如何开启这些开关
- 方法一：修改 `config.py -> DEFAULT_CONFIG -> system` 中对应字段设为 `True`，或在预设中覆盖（`PRESETS[...] -> system`）。
- 方法二：复制一个预设，直接在该预设的 `system` 字段中将需要的开关设为 `True`。
> 当前 CLI 未提供独立开关参数，`--method` 仅决定执行模式，自适应/注入需通过 `config.py` 调整。

## 自适应实验（adaptive_experiment）关键参数
```python
@dataclass
class AdaptiveExperimentConfig:
    total_epochs: int = 1000
    adaptive_cycle_epochs: int = 200
    detect_every: int = 100
    num_residual_scout_points: int = 5000
    exploration_initial: float = 0.2
    exploration_final: float = 0.05
    exploration_decay: float = 0.02
    enable_kriging: bool = True
    enable_data_injection: bool = False
    enable_rapid_improvement_early_stop: bool = True
    split_ratios: list = field(default_factory=lambda:[0.7,0.05,0.05,0.05,0.05,0.05,0.05])
    test_set_size: int = 300
    enable_baseline: bool = True
    file_suffix: str = "full_adaptive"
```

## PINN 训练日志 / 回退 / 早停参数
```python
@dataclass
class PinnTrainingParams:
    total_epochs: int = 5000
    cycle_epochs: int = 5000
    adaptive_cycle_epochs: int = 2000
    detect_every: int = 500
    detection_threshold: float = 0.1  # 早停/快速改善阈值，透传 run_training_cycle
```
机制说明：`EarlyCycleStopper` 在检测间隔内监控 MRE，保存最佳；若性能变差则回滚并提前结束当前周期；快速改善超过阈值也会提前结束。

## 事件标记（训练曲线）
- 当开启自适应或回退触发时，事件会写入训练历史并在输出图上画竖线标注：
  - `phase_transition`: 首轮结束 / 第二轮结束。
  - `kriging_resampling`: Compose 模式残差引导配点完成。
  - `early_stop`: 早停/回退触发（当 `stagnation_detected=True`）。

