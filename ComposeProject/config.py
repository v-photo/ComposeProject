"""
PINN-Kriging 耦合系统配置文件
Configuration file for PINN-Kriging coupling system

使用方法：
1. 修改本文件中的参数
2. 运行主程序时会自动加载这些配置
3. 无需修改源代码即可调整实验参数
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

# ==================== 全局实验控制 ====================
@dataclass
class ExperimentConfig:
    """实验模式控制配置"""
    # 主要策略开关
    enable_kriging: bool = True              # 是否启用克里金引导重采样
    enable_data_injection: bool = False      # 是否启用数据注入策略  
    enable_rapid_improvement_early_stop: bool = False  # 快速改善早停
    
    # 实验标识（用于文件命名和日志）
    experiment_name: str = "adaptive_pinn"   # 实验名称
    
    def get_experiment_type(self) -> str:
        """根据开关组合返回实验类型描述"""
        if self.enable_kriging and self.enable_data_injection:
            return "完整自适应PINN"
        elif self.enable_kriging and not self.enable_data_injection:
            return "仅克里金重采样"
        elif not self.enable_kriging and self.enable_data_injection:
            return "仅数据注入"
        else:
            return "基线对比"

# ==================== 数据加载配置 ====================
@dataclass
class DataConfig:
    """数据加载相关配置"""
    # 数据文件路径
    data_path: str = "PINN/DATA.xlsx"
    
    # 物理空间尺寸 [x, y, z] (米)
    space_dims: List[float] = None
    
    # 采样配置
    num_samples: int = 100                   # 初始训练样本数
    test_set_size: int = 300                 # 独立测试集大小
    
    # 数据分割比例 [主训练集, 储备池1, 储备池2, ...]
    # 剩余部分自动作为测试集
    data_split_ratios: List[float] = None
    
    def __post_init__(self):
        if self.space_dims is None:
            self.space_dims = [20.0, 10.0, 10.0]
        if self.data_split_ratios is None:
            self.data_split_ratios = [0.5] + [0.1] * 5

# ==================== PINN训练配置 ====================  
@dataclass
class PINNConfig:
    """PINN模型训练配置"""
    # 网络结构
    network_layers: List[int] = None         # 神经网络层结构
    activation: str = "tanh"                 # 激活函数
    
    # 训练参数
    total_epochs: int = 8000                 # 总训练轮数
    adaptive_cycle_epochs: int = 2000        # 每个自适应周期的轮数
    detect_epochs: int = 500                 # 性能检测间隔
    learning_rate: float = 1e-3              # 学习率
    
    # 配置点设置
    num_collocation_points: int = 4096       # 求解域配置点数量
    num_residual_scout_points: int = 5000    # 残差侦察点数量
    
    # 损失权重策略
    use_dynamic_loss_strategy: bool = True   # 是否使用动态损失权重
    initial_loss_ratio: float = 10.0        # 初始数据/物理损失比值
    final_loss_ratio: float = 0.1           # 最终数据/物理损失比值
    fixed_loss_ratio: float = 10.0          # 固定策略时的比值
    
    # 物理参数（示例值，需根据具体问题调整）
    physical_params: Dict[str, float] = None
    
    def __post_init__(self):
        if self.network_layers is None:
            self.network_layers = [3, 64, 64, 64, 1]
        if self.physical_params is None:
            self.physical_params = {
                'rho_material': 1.2,               # 材料密度
                'mass_energy_abs_coeff': 1.0,      # 质量能量吸收系数
                'rho': 1.2,                        # 通用密度参数
                'mu': 1e-3                         # 粘度
            }

# ==================== 克里金配置 ====================
@dataclass  
class KrigingConfig:
    """克里金模型配置"""
    # 模型参数
    variogram_model: str = "exponential"     # 变异函数模型
    nlags: int = 8                          # 滞后数
    block_size: int = 10000                 # GPU处理块大小
    
    # 自适应采样策略
    initial_exploration_ratio: float = 0.50  # 初始探索率
    final_exploration_ratio: float = 0.18   # 最终探索率  
    exploration_decay_rate: float = 0.03    # 每周期探索率衰减
    total_candidates: int = 100000          # 候选点池大小
    
    # 数据分布分析阈值
    uniformity_cv_threshold: float = 0.6    # 最近邻距离变异系数阈值

# ==================== 系统配置 ====================
@dataclass
class SystemConfig:
    """系统运行配置"""
    # GPU加速
    gpu_enabled: bool = True                 # 是否启用GPU加速
    
    # 随机种子
    random_seed: int = 42                    # 随机种子，确保结果可复现
    
    # 日志和输出
    verbose: bool = True                     # 是否输出详细日志
    save_results: bool = True                # 是否保存结果文件
    results_dir: str = "results"             # 结果保存目录
    
    # 可视化
    figure_dpi: int = 300                    # 图片分辨率
    figure_format: List[str] = None          # 保存格式
    
    def __post_init__(self):
        if self.figure_format is None:
            self.figure_format = ["png", "pdf"]

# ==================== 主配置类 ====================
@dataclass
class Config:
    """主配置类，整合所有子配置"""
    experiment: ExperimentConfig = None
    data: DataConfig = None  
    pinn: PINNConfig = None
    kriging: KrigingConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        # 如果子配置为None，则使用默认值
        if self.experiment is None:
            self.experiment = ExperimentConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.pinn is None:
            self.pinn = PINNConfig()
        if self.kriging is None:
            self.kriging = KrigingConfig()
        if self.system is None:
            self.system = SystemConfig()
    
    def summary(self) -> str:
        """返回配置摘要"""
        summary = []
        summary.append("="*60)
        summary.append("🔧 PINN-Kriging 耦合系统配置摘要")
        summary.append("="*60)
        
        # 实验配置
        summary.append(f"📋 实验类型: {self.experiment.get_experiment_type()}")
        summary.append(f"   - 克里金重采样: {'✅' if self.experiment.enable_kriging else '❌'}")
        summary.append(f"   - 数据注入: {'✅' if self.experiment.enable_data_injection else '❌'}")
        summary.append(f"   - 快速改善早停: {'✅' if self.experiment.enable_rapid_improvement_early_stop else '❌'}")
        
        # 训练配置
        summary.append(f"🧠 PINN训练: {self.pinn.total_epochs}轮 (每{self.pinn.adaptive_cycle_epochs}轮自适应)")
        summary.append(f"   - 网络结构: {self.pinn.network_layers}")
        summary.append(f"   - 损失策略: {'动态' if self.pinn.use_dynamic_loss_strategy else '固定'}")
        if self.pinn.use_dynamic_loss_strategy:
            summary.append(f"     └─ 比值变化: {self.pinn.initial_loss_ratio:.1f} → {self.pinn.final_loss_ratio:.1f}")
        else:
            summary.append(f"     └─ 固定比值: {self.pinn.fixed_loss_ratio:.1f}")
        
        # 数据配置  
        summary.append(f"📊 数据配置: {self.data.num_samples}个训练样本, {self.data.test_set_size}个测试样本")
        summary.append(f"   - 物理空间: {self.data.space_dims} (米)")
        
        # 克里金配置
        if self.experiment.enable_kriging:
            summary.append(f"🗺️  克里金配置: {self.kriging.variogram_model}变异函数")
            summary.append(f"   - 探索率: {self.kriging.initial_exploration_ratio:.1%} → {self.kriging.final_exploration_ratio:.1%}")
        
        # 系统配置
        summary.append(f"⚙️  系统配置: GPU{'启用' if self.system.gpu_enabled else '禁用'}, 随机种子={self.system.random_seed}")
        
        summary.append("="*60)
        return "\n".join(summary)

# ==================== 预设配置 ====================
def get_preset_config(preset_name: str) -> Config:
    """获取预设配置
    
    Args:
        preset_name: 预设名称
            - "full_adaptive": 完整自适应策略
            - "kriging_only": 仅克里金重采样  
            - "data_injection_only": 仅数据注入
            - "baseline": 基线对比
            - "quick_test": 快速测试（小规模）
    """
    base_config = Config()
    
    if preset_name == "full_adaptive":
        base_config.experiment.enable_kriging = True
        base_config.experiment.enable_data_injection = True
        base_config.experiment.experiment_name = "full_adaptive"
        
    elif preset_name == "kriging_only":
        base_config.experiment.enable_kriging = True
        base_config.experiment.enable_data_injection = False
        base_config.experiment.experiment_name = "kriging_only"
        
    elif preset_name == "data_injection_only":
        base_config.experiment.enable_kriging = False
        base_config.experiment.enable_data_injection = True
        base_config.experiment.experiment_name = "data_injection_only"
        
    elif preset_name == "baseline":
        base_config.experiment.enable_kriging = False
        base_config.experiment.enable_data_injection = False
        base_config.experiment.experiment_name = "baseline"
        
    elif preset_name == "quick_test":
        base_config.experiment.enable_kriging = True
        base_config.experiment.enable_data_injection = False
        base_config.experiment.experiment_name = "quick_test"
        # 快速测试配置
        base_config.pinn.total_epochs = 1000
        base_config.pinn.adaptive_cycle_epochs = 500
        base_config.data.num_samples = 50
        base_config.data.test_set_size = 100
        base_config.pinn.num_collocation_points = 1024
        
    else:
        raise ValueError(f"未知的预设配置: {preset_name}")
    
    return base_config

# ==================== 默认配置实例 ====================
# 创建默认配置实例，供外部导入使用
default_config = Config()

if __name__ == "__main__":
    # 演示配置使用
    print("🔧 配置文件演示")
    print(default_config.summary())
    
    print("\n" + "="*40)
    print("📋 可用预设配置:")
    presets = ["full_adaptive", "kriging_only", "data_injection_only", "baseline", "quick_test"]
    for preset in presets:
        config = get_preset_config(preset)
        print(f"  - {preset}: {config.experiment.get_experiment_type()}") 