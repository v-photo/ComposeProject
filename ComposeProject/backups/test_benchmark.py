import sys
import os
import matplotlib.pyplot as plt

def main():
    """
    一个临时的、极简的主函数，其唯一目的是
    调用和测试 run_pinn_benchmark 函数。
    """
    # 将项目根目录添加到 sys.path
    # __file__ -> test_benchmark.py
    # os.path.dirname -> ComposeProject/
    # .parent -> PINN_ts/ (根)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    # 现在路径设置好了，可以安全导入了
    from ComposeTools import run_pinn_benchmark

    # 直接运行基准测试
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python 路径: {sys.path}")
    
    run_pinn_benchmark(epochs=2000, show_plots=True)


if __name__ == "__main__":
    main() 