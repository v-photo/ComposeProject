"""
分析与绘图模块
Module for analysis and plotting.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

# 尝试设置中文字体，如果失败则回退
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("警告: 无法设置中文字体，图表可能无法正确显示字符。")


def plot_training_comparison(
    models_history: Dict[str, Dict[str, np.ndarray]],
    important_events: Optional[List[Tuple[int, str, str]]] = None,
    title: str = "模型训练历史对比",
    save_path: Optional[str] = None):
    """
    绘制多个模型训练历史（如MRE）的对比图，并能高亮标注训练过程中的重要事件。

    Args:
        models_history (Dict): 一个字典，键是模型名称，值是包含 'epochs' 和 'metrics' 的字典。
                               例如: {'自适应PINN': {'epochs': [...], 'metrics': [...]}}
        important_events (List, optional): 一个包含重要事件的列表，每个事件是一个元组
                                           (epoch, event_type, description)。
        title (str): 图表的标题。
        save_path (str, optional): 如果提供，则将图表保存到指定路径，而不是显示它。
    """
    print("\n" + "="*60)
    print(f"📈 正在生成图表: {title}")
    print("="*60 + "\n")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # 绘制每个模型的历史曲线
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_history)))
    for i, (name, history) in enumerate(models_history.items()):
        if 'epochs' in history and 'metrics' in history and len(history['epochs']) > 0:
            ax.plot(history['epochs'], history['metrics'], 
                    label=name, linewidth=2, alpha=0.8, color=colors[i])

    # 绘制重要事件的标注
    if important_events:
        _plot_smart_annotations(ax, important_events)

    # 设置图表样式
    ax.set_xlabel("训练轮数 (Epochs)", fontsize=14)
    ax.set_ylabel("平均相对误差 (MRE)", fontsize=14)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=18, weight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 合并主图例和事件图例
    handles, labels = ax.get_legend_handles_labels()
    
    event_legend_handles = ax.get_legend()
    if event_legend_handles:
        handles.extend(event_legend_handles.legendHandles)
        labels.extend([text.get_text() for text in event_legend_handles.texts])
        event_legend_handles.remove()

    ax.legend(handles, labels, fontsize=12, loc='lower left')

    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
        print(f"  ✅ 图表已保存至: {p}")
    else:
        # 如果没有提供保存路径，则显示图表
        plt.show()
    
    plt.close(fig) # 释放内存


def _plot_smart_annotations(ax: plt.Axes, events: List[Tuple[int, str, str]]):
    """在图表上使用智能算法绘制事件标注，以避免文本重叠。"""

    def _get_smart_positions(sorted_events, y_range):
        """智能计算标注的Y轴位置。"""
        if not sorted_events: return []
        
        y_min, y_max = y_range
        positions = []

        # 在 log 区间内均匀取若干高度，避免全部落在同一条线上
        log_min, log_max = np.log10(max(y_min, 1e-10)), np.log10(max(y_max, 1e-9))
        span = max(log_max - log_min, 1e-6)
        start_exp = log_max - 0.1 * span      # 接近顶部但预留空间
        end_exp = log_max - 0.6 * span        # 向下分布避免重叠
        y_levels = np.logspace(start_exp, end_exp, 8)

        x_range = sorted_events[-1][0] - sorted_events[0][0] if len(sorted_events) > 1 else 1
        min_distance = max(200, x_range * 0.05)

        occupied = [] # (epoch, level_index)
        for epoch, _, _ in sorted_events:
            best_level = 0
            min_conflicts = float('inf')
            for level_idx in range(len(y_levels)):
                conflicts = sum(1 for prev_epoch, prev_level in occupied if abs(epoch - prev_epoch) < min_distance and level_idx == prev_level)
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_level = level_idx
                if conflicts == 0: break
            occupied.append((epoch, best_level))
        
        return [(event[0], y_levels[pos[1]]) for event, pos in zip(sorted_events, occupied)]

    event_styles = {
        'data_injection': {'color': 'green', 'linestyle': '--', 'label': '数据注入'},
        'kriging_resampling': {'color': 'orange', 'linestyle': '-.', 'label': '克里金重采样'},
        'rollback': {'color': 'purple', 'linestyle': ':', 'label': '回退最佳模型'},
        'loss_ratio_update': {'color': 'red', 'linestyle': '-', 'label': '权重更新'}
    }

    sorted_events = sorted(events, key=lambda x: x[0])
    y_min, y_max = ax.get_ylim()
    annotation_positions = _get_smart_positions(sorted_events, (y_min, y_max))

    legend_handles = {}
    type_counters = {}
    for i, (event_data, pos_data) in enumerate(zip(sorted_events, annotation_positions)):
        epoch, event_type, description = event_data
        y_pos = pos_data[1]
        # 同类型事件分层偏移，减少遮挡
        type_counters.setdefault(event_type, 0)
        offset_factor = 1 + 0.12 * (type_counters[event_type] % 3)  # 同一类型最多轮换3层
        type_counters[event_type] += 1
        y_pos *= offset_factor

        style = event_styles.get(event_type, {'color': 'gray', 'linestyle': '-', 'label': '其他'})

        line = ax.axvline(x=epoch, color=style['color'], linestyle=style['linestyle'], alpha=0.7, linewidth=1.5)
        if style['label'] not in legend_handles:
            legend_handles[style['label']] = line
        
        short_desc = description.split('(')[0].strip()[:30]
        ax.annotate(f'{short_desc}\n(E{epoch})',
                    xy=(epoch, y_pos), xytext=(8, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=style['color'], alpha=0.8),
                    arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.1',
                                  color=style['color'], alpha=0.6))

    # 单独放置事件图例，避免与主图例重叠
    ax.legend(handles=legend_handles.values(), labels=legend_handles.keys(),
              title="重要事件", fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
