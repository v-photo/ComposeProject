#!/usr/bin/env python3
"""
CUDA诊断和测试脚本
CUDA Diagnostic and Testing Script

用于诊断和修复CUDA相关问题
"""

import os
import sys
import numpy as np
import warnings
import time

def test_basic_cuda():
    """测试基础CUDA功能"""
    print("🔍 测试基础CUDA功能...")
    
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        print(f"   CUDA版本: {torch.version.cuda}")
        
        if torch.cuda.is_available():
            print(f"   GPU设备数量: {torch.cuda.device_count()}")
            print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            
            # 测试简单的CUDA运算
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda() 
            z = torch.matmul(x, y)
            print(f"   ✅ CUDA矩阵运算测试成功")
            
            return True
        else:
            print("   ❌ CUDA不可用")
            return False
            
    except Exception as e:
        print(f"   ❌ CUDA测试失败: {e}")
        return False

def test_deepxde_cuda():
    """测试DeepXDE的CUDA配置"""
    print("🔍 测试DeepXDE CUDA配置...")
    
    try:
        import deepxde as dde
        print(f"   DeepXDE版本: {dde.__version__}")
        print(f"   后端: {dde.backend.backend_name}")
        
        # 测试简单的网络创建
        import torch
        torch.set_default_device('cuda')
        
        # 创建简单的数据
        X = np.random.randn(50, 2)
        y = np.sum(X**2, axis=1, keepdims=True)
        
        # 创建几何和网络
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)
            return dy_xx + dy_yy
        
        def boundary(x, on_boundary):
            return on_boundary
        
        def func(x):
            return np.zeros((len(x), 1))
        
        bc = dde.icbc.DirichletBC(geom, func, boundary)
        data = dde.data.PDE(geom, pde, bc, num_domain=100, num_boundary=50)
        
        net = dde.nn.FNN([2, 20, 20, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        
        print("   ✅ DeepXDE网络创建成功")
        
        # 测试短期训练
        model.compile("adam", lr=1e-3)
        model.train(iterations=10, display_every=10)
        
        print("   ✅ DeepXDE训练测试成功")
        return True
        
    except Exception as e:
        print(f"   ❌ DeepXDE CUDA测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpu_fallback():
    """测试CPU回退机制"""
    print("🔍 测试CPU回退机制...")
    
    try:
        import torch
        # 强制使用CPU
        torch.set_default_device('cpu')
        
        import deepxde as dde
        
        # 创建相同的测试
        X = np.random.randn(50, 2)
        y = np.sum(X**2, axis=1, keepdims=True)
        
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            dy_yy = dde.grad.hessian(y, x, i=1, j=1)
            return dy_xx + dy_yy
        
        def boundary(x, on_boundary):
            return on_boundary
        
        def func(x):
            return np.zeros((len(x), 1))
        
        bc = dde.icbc.DirichletBC(geom, func, boundary)
        data = dde.data.PDE(geom, num_domain=100, num_boundary=50, pde=pde, bcs=[bc])
        
        net = dde.nn.FNN([2, 20, 20, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        
        model.compile("adam", lr=1e-3)
        model.train(iterations=10, display_every=10)
        
        print("   ✅ CPU模式测试成功")
        return True
        
    except Exception as e:
        print(f"   ❌ CPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def apply_cuda_fixes():
    """应用CUDA修复方案"""
    print("🔧 应用CUDA修复方案...")
    
    fixes_applied = []
    
    # 修复1: 设置环境变量
    try:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        fixes_applied.append("设置CUDA_LAUNCH_BLOCKING=1")
    except:
        pass
    
    # 修复2: 清理GPU缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            fixes_applied.append("清理GPU缓存")
    except:
        pass
    
    # 修复3: 重置默认设备
    try:
        import torch
        torch.set_default_device('cpu')  # 先设为CPU
        if torch.cuda.is_available():
            torch.set_default_device('cuda')  # 再设为CUDA
            fixes_applied.append("重置设备状态")
    except:
        pass
    
    for fix in fixes_applied:
        print(f"   ✅ {fix}")
    
    return len(fixes_applied) > 0

def main():
    """主函数"""
    print("🚀 CUDA诊断和修复工具")
    print("=" * 50)
    
    # 基础CUDA测试
    basic_cuda_ok = test_basic_cuda()
    print()
    
    # 应用修复
    if basic_cuda_ok:
        fixes_applied = apply_cuda_fixes()
        print()
        
        # DeepXDE CUDA测试
        deepxde_cuda_ok = test_deepxde_cuda()
        print()
        
        if not deepxde_cuda_ok:
            print("⚠️ CUDA模式失败，测试CPU回退...")
            cpu_ok = test_cpu_fallback()
            print()
            
            if cpu_ok:
                print("💡 建议:")
                print("   1. 使用CPU模式运行: python main.py --mode mode1 --no_gpu")
                print("   2. 或设置环境变量: CUDA_VISIBLE_DEVICES='' python main.py --mode mode1")
                print("   3. 代码已添加自动CUDA错误恢复机制")
            else:
                print("❌ CPU模式也失败，请检查DeepXDE安装")
        else:
            print("✅ CUDA修复成功！可以正常使用GPU加速")
    else:
        print("❌ 基础CUDA不可用，建议:")
        print("   1. 检查CUDA驱动安装")
        print("   2. 检查PyTorch CUDA版本匹配")
        print("   3. 使用CPU模式: python main.py --mode mode1 --no_gpu")

if __name__ == "__main__":
    main() 