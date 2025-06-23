#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探索率配置测试脚本
用于可视化不同探索率策略的变化曲线，帮助选择合适的参数配置
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_exploration_ratio(cycle_number, initial_ratio, final_ratio, decay_rate):
    """
    计算指定周期的探索率
    """
    return max(
        final_ratio,
        initial_ratio - (cycle_number - 1) * decay_rate
    )

def plot_exploration_strategies():
    """
    绘制不同探索率策略的变化曲线
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 定义周期范围
    cycles = np.arange(1, 13)  # 12个自适应周期
    
    # 定义不同的策略配置
    strategies = {
        '当前配置 (适中策略)': {
            'initial': 0.20, 'final': 0.05, 'decay': 0.02,
            'color': 'blue', 'linestyle': '-'
        },
        '激进策略 (快速收敛)': {
            'initial': 0.25, 'final': 0.02, 'decay': 0.05,
            'color': 'red', 'linestyle': '--'
        },
        '保守策略 (长期探索)': {
            'initial': 0.15, 'final': 0.08, 'decay': 0.01,
            'color': 'green', 'linestyle': '-.'
        },
        '精准策略 (高利用率)': {
            'initial': 0.30, 'final': 0.03, 'decay': 0.03,
            'color': 'orange', 'linestyle': ':'
        }
    }
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制探索率变化曲线
    for strategy_name, config in strategies.items():
        exploration_ratios = [
            calculate_exploration_ratio(cycle, config['initial'], config['final'], config['decay'])
            for cycle in cycles
        ]
        
        ax1.plot(cycles, exploration_ratios, 
                label=strategy_name, 
                color=config['color'], 
                linestyle=config['linestyle'],
                linewidth=2, marker='o', markersize=6)
        
        # 计算利用率 (1 - 探索率) 用于第二个子图
        exploitation_ratios = [1 - ratio for ratio in exploration_ratios]
        ax2.plot(cycles, exploitation_ratios,
                label=strategy_name,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=2, marker='s', markersize=6)
    
    # 设置第一个子图 (探索率)
    ax1.set_xlabel('自适应周期', fontsize=12)
    ax1.set_ylabel('探索率 (%)', fontsize=12)
    ax1.set_title('🔍 不同策略的探索率变化曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.35)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 设置第二个子图 (利用率)
    ax2.set_xlabel('自适应周期', fontsize=12)
    ax2.set_ylabel('利用率 (%)', fontsize=12)
    ax2.set_title('🎯 不同策略的利用率变化曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.65, 1.0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'exploration_rate_strategies.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 探索率策略对比图已保存到: {output_path}")
    
    plt.show()

def print_strategy_details():
    """
    打印各策略的详细数值变化
    """
    print("\n" + "="*80)
    print("📊 各策略的探索率变化详情")
    print("="*80)
    
    strategies = {
        '当前配置 (适中策略)': (0.20, 0.05, 0.02),
        '激进策略 (快速收敛)': (0.25, 0.02, 0.05),
        '保守策略 (长期探索)': (0.15, 0.08, 0.01),
        '精准策略 (高利用率)': (0.30, 0.03, 0.03)
    }
    
    cycles = range(1, 13)
    
    for strategy_name, (initial, final, decay) in strategies.items():
        print(f"\n🎯 {strategy_name}")
        print(f"   配置: 初始={initial:.0%}, 最终={final:.0%}, 递减={decay:.0%}/周期")
        print("   周期变化:", end=" ")
        
        for cycle in cycles:
            exploration_ratio = calculate_exploration_ratio(cycle, initial, final, decay)
            print(f"第{cycle}周期:{exploration_ratio:.0%}", end="  ")
            if cycle % 4 == 0:  # 每4个周期换行
                print("\n              ", end=" ")
        print()  # 结束换行
    
    print("\n" + "="*80)
    print("💡 选择建议:")
    print("   - 数据稀疏/复杂问题: 建议使用保守策略，保持较多探索")
    print("   - 训练时间有限: 建议使用激进策略，快速聚焦到高残差区域")
    print("   - 一般情况: 当前适中策略平衡探索与利用")
    print("   - 已知问题特性: 可使用精准策略，早期大量探索后快速收敛")
    print("="*80)

if __name__ == "__main__":
    print("🚀 探索率配置分析工具")
    print("="*50)
    
    # 绘制策略对比图
    plot_exploration_strategies()
    
    # 打印详细数值
    print_strategy_details()
    
    print("\n🔧 修改探索率参数:")
    print("   在 example2.py 文件的第41-43行修改以下参数:")
    print("   - INITIAL_EXPLORATION_RATIO: 初始探索率")
    print("   - FINAL_EXPLORATION_RATIO: 最终探索率")
    print("   - EXPLORATION_DECAY_RATE: 每周期递减率") 