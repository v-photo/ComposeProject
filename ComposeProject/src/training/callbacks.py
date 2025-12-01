"""
训练过程中的自定义回调函数
Custom callback functions for the training process.
"""
import os
import numpy as np
import deepxde as dde

class EarlyCycleStopper(dde.callbacks.Callback):
    """
    一个自定义回调，用于在训练周期内实现基于性能的"早停"。
    同时自己负责保存周期内的最佳模型。
    """
    def __init__(self, detection_threshold: float, display_every: int, checkpoint_path_prefix: str):
        super().__init__()
        self.threshold = detection_threshold
        self.display_every = display_every
        self.checkpoint_path_prefix = checkpoint_path_prefix
        self.best_mre = np.inf
        self.should_stop = False
        self.best_model_path = "" # 将存储最佳模型的完整真实路径

    def reset_cycle(self, initial_mre: float = np.inf, initial_model_path: str = ""):
        """
        手动重置整个周期的状态，为新的自适应周期做准备。
        可以接收一个初始MRE和模型路径作为本周期的性能基线。
        """
        # 清理上一轮可能遗留的检查点文件
        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
        
        self.best_mre = initial_mre
        self.best_model_path = initial_model_path
        self.should_stop = False

    def on_epoch_end(self):
        """在每个 epoch 结束时被调用, 并且在这里检查性能"""
        # 只有在达到指定的检测间隔时才进行检查
        if self.model.train_state.step > 0 and self.model.train_state.step % self.display_every == 0:
            if not self.model.train_state.metrics_test:
                 return

            latest_mre = self.model.train_state.metrics_test[-1]
            
            # 检查是否有显著的性能提升
            # 仅在 self.best_mre 不是无穷大（即至少有一个基准）时检查
            if self.best_mre != np.inf:
                improvement = self.best_mre - latest_mre
                required_improvement_amount = self.best_mre * self.threshold
                
                # 如果性能提升超过阈值，则标记为可以停止
                if improvement > required_improvement_amount:
                    print(f"    💡 Early Stop: MRE dropped from {self.best_mre:.6f} to {latest_mre:.6f} (>{self.threshold:.0%}).")
                    self.should_stop = True
            
            # 判断当前模型是否是新的周期内最佳模型
            if latest_mre < self.best_mre:
                print(f"    ⭐ New best model in cycle (MRE: {latest_mre:.6f}). Checkpointing...")
                self.best_mre = latest_mre

                # 为了防止文件残留，先清理上一个最佳模型
                if self.best_model_path and os.path.exists(self.best_model_path):
                    try:
                        os.remove(self.best_model_path)
                    except OSError as e:
                        print(f"Warning: Could not remove previous best model file: {e}")
                
                # 构建新的最佳模型路径并保存
                # 注意：DDE的save方法会自动在路径后添加-step.pt
                # 所以我们只需提供前缀
                self.model.save(self.checkpoint_path_prefix, verbose=0)
                # 更新 best_model_path 以便下次可以清理
                current_step = self.model.train_state.step
                self.best_model_path = f"{self.checkpoint_path_prefix}-{current_step}.pt"
