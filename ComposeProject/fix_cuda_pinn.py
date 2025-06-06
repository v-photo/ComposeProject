#!/usr/bin/env python3
"""
专门修复PINN CUDA问题的诊断和修复脚本
PINN CUDA Problem Diagnostic and Fix Script
"""

import os
import sys
import warnings
import numpy as np

def fix_cuda_context():
    """修复CUDA上下文和设备状态"""
    print("🔧 修复CUDA上下文...")
    
    try:
        import torch
        
        # 1. 强制清理CUDA状态
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            print("   ✅ CUDA缓存已清理")
        
        # 2. 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        print("   ✅ CUDA内存分配策略已设置")
        
        # 3. 设置CUDA错误检查
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("   ✅ CUDA启动阻塞已启用")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CUDA上下文修复失败: {e}")
        return False

def test_deepxde_cuda_minimal():
    """测试最小的DeepXDE CUDA操作"""
    print("🧪 测试最小DeepXDE CUDA操作...")
    
    try:
        import torch
        import deepxde as dde
        
        # 强制设置为CPU先测试
        torch.set_default_device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
        
        # 创建最简单的问题
        geom = dde.geometry.Interval(0, 1)
        
        def pde(x, y):
            dy_xx = dde.grad.hessian(y, x)
            return dy_xx + 1
        
        def boundary(x, on_boundary):
            return on_boundary
        
        def func(x):
            return np.zeros((len(x), 1))
        
        bc = dde.icbc.DirichletBC(geom, func, boundary)
        data = dde.data.PDE(geom, pde, bc, num_domain=50, num_boundary=2)
        
        net = dde.nn.FNN([1, 10, 1], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        
        # 先在CPU上编译和训练
        model.compile("adam", lr=1e-3)
        model.train(iterations=5, display_every=5)
        
        print("   ✅ CPU模式DeepXDE测试成功")
        
        # 现在尝试CUDA
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            # 重新创建模型用于CUDA
            net_cuda = dde.nn.FNN([1, 10, 1], "tanh", "Glorot normal")
            model_cuda = dde.Model(data, net_cuda)
            
            model_cuda.compile("adam", lr=1e-3)
            model_cuda.train(iterations=5, display_every=5)
            
            print("   ✅ CUDA模式DeepXDE测试成功")
            return True
        else:
            print("   ⚠️ CUDA不可用，无法测试CUDA模式")
            return False
            
    except Exception as e:
        print(f"   ❌ DeepXDE CUDA测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def apply_comprehensive_fix():
    """应用综合修复方案"""
    print("🚀 应用PINN CUDA综合修复方案")
    print("=" * 50)
    
    # 1. 修复CUDA上下文
    cuda_fixed = fix_cuda_context()
    
    # 2. 测试DeepXDE
    if cuda_fixed:
        deepxde_ok = test_deepxde_cuda_minimal()
        
        if deepxde_ok:
            print("\n✅ 修复成功！现在可以运行GPU模式")
            return True
        else:
            print("\n❌ DeepXDE CUDA仍有问题")
            return False
    else:
        print("\n❌ CUDA基础修复失败")
        return False

def suggest_alternative_solutions():
    """建议其他解决方案"""
    print("\n💡 其他可能的解决方案:")
    print("1. 数据类型问题修复:")
    print("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
    print("   export CUDA_LAUNCH_BLOCKING=1")
    print()
    print("2. 网络配置调整:")
    print("   - 减小网络层数: [3, 16, 16, 1]")
    print("   - 使用更小的批次大小")
    print("   - 减少训练样本数量")
    print()
    print("3. PyTorch版本问题:")
    print("   可能需要重新安装PyTorch和CUDA")
    print("   pip uninstall torch")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    success = apply_comprehensive_fix()
    
    if not success:
        suggest_alternative_solutions()
        sys.exit(1)
    else:
        print("\n🎉 修复完成！可以继续运行GPU模式的PINN训练") 