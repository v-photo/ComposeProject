import sys
import os

# --- 核心：只导入我们需要的那个可靠的函数 ---
from ComposeTools import run_pinn_benchmark

def main():
    """
    一个临时的、极简的主函数，其唯一目的是
    调用和测试 run_pinn_benchmark 函数。
    """
    # 直接运行基准测试
    run_pinn_benchmark(epochs=2000, show_plots=True)

if __name__ == "__main__":
    # 为了让 run_pinn_benchmark 中的相对路径能工作，
    # 我们需要确保当前工作目录是项目的根目录。
    # 这个脚本位于 ComposeProject/main.py
    # 我们需要 cd .. 回到项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.chdir(project_root)
    
    # 将项目根目录添加到 sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 调用主函数
    main()