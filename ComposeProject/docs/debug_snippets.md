# 调试代码片段记录

本文档记录了在 `GPU Block-Kriging × PINN` 项目开发过程中使用过的调试代码片段，以便未来查阅和复用。

---

## 1. PINN 残差三维空间分布可视化

- **功能**:
  将 PINN 在训练点上产生的残差（真实值 - PINN预测值）以三维散点图的形式可视化。散点的颜色代表残差的大小。这有助于直观地判断残差是否存在空间聚集性或特定模式，从而为 Kriging 插值提供依据。

- **插入位置**:
  - **文件**: `ComposeProject/ComposeTools.py`
  - **类**: `Mode1ResidualKriging`
  - **方法**: `residual_kriging`
  - **具体位置**: 在计算完 `residuals` 变量之后，训练Kriging模型之前。

- **代码片段**:
```python
# ==================== 纯调试代码: 分析输入的残差 ====================
try:
    print("\n" + "-"*20 + " DEBUG: Residual Analysis " + "-"*20)
    print(f"Residuals shape: {residuals.shape}, dtype: {residuals.dtype}")
    print(f"Contains NaNs: {np.isnan(residuals).any()}, Contains Infs: {np.isinf(residuals).any()}")
    print(f"Stats: Mean={np.mean(residuals):.4e}, Std={np.std(residuals):.4e}")
    print(f"Stats: Min={np.min(residuals):.4e}, Median={np.median(residuals):.4e}, Max={np.max(residuals):.4e}")
    
    # 可视化残差的三维空间分布
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(train_points[:, 0], train_points[:, 1], train_points[:, 2], 
                         c=residuals, cmap='viridis', s=40, alpha=0.8)
    
    ax.set_title("3D Spatial Distribution of Residuals for Kriging")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Z coordinate")
    
    cbar = fig.colorbar(scatter, shrink=0.7, aspect=20)
    cbar.set_label("Residual Value (True - PINN Pred)")
    
    debug_plot_path = "debug_residual_3d_distribution.png"
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Residual 3D distribution plot saved to: {debug_plot_path}")
    print("-"*(40 + len(" DEBUG: Residual Analysis ")) + "\n")

except Exception as e:
    print(f"❌ DEBUG: Failed to visualize residuals: {e}")
# ========================== 调试代码结束 ==========================
```

---

## 2. Kriging 对训练残差的自预测精度测试

- **功能**:
  使用刚刚在训练集残差上训练好的 Kriging 模型，立即对这些训练点本身进行一次"回测"或"自预测"。通过计算其预测值与真实残差之间的平均相对误差（MRE），来评估 Kriging 模型对训练数据的拟合程度。这可以帮助判断 Kriging 是否有效学习到了残差的空间结构，或者是否存在过拟合/欠拟合问题。

- **插入位置**:
  - **文件**: `ComposeProject/ComposeTools.py`
  - **类**: `Mode1ResidualKriging`
  - **方法**: `residual_kriging`
  - **具体位置**: 在调用 `self.kriging_adapter.fit()` 训练完模型之后，进行真实预测之前。

- **代码片段**:
```python
# ==================== 纯调试代码: 测试Kriging对训练残差的自预测精度 ====================
try:
    print("\n" + "-"*20 + " DEBUG: Kriging Self-Prediction Test " + "-"*20)
    # 使用刚刚训练好的模型，在训练点上进行预测
    kriging_train_pred = self.kriging_adapter.predict(train_points, return_std=False)
    
    # 计算Kriging预测值与真实残差之间的平均相对误差 (MRE)
    abs_true_residuals = np.abs(residuals)
    
    # 避免除以零
    valid_mask = abs_true_residuals > 1e-30 # Using a small epsilon
    if np.any(valid_mask):
        relative_errors = np.abs(residuals[valid_mask] - kriging_train_pred[valid_mask]) / abs_true_residuals[valid_mask]
        mre = np.mean(relative_errors)
        print(f"✅ Kriging on Training Residuals MRE: {mre:.6f}")
        
        # 额外提供一些统计
        print(f"  - Test points count: {np.sum(valid_mask)}")
        print(f"  - True Residuals (on test points): Mean={np.mean(residuals[valid_mask]):.4e}, Std={np.std(residuals[valid_mask]):.4e}")
        print(f"  - Predicted Residuals (on test points): Mean={np.mean(kriging_train_pred[valid_mask]):.4e}, Std={np.std(kriging_train_pred[valid_mask]):.4e}")

    else:
        print("⚠️ Kriging MRE test skipped: All true residual values are close to zero.")

    print("-"*(40 + len(" DEBUG: Kriging Self-Prediction Test ")) + "\n")
except Exception as e:
    print(f"❌ DEBUG: Failed to test Kriging self-prediction: {e}")
# ========================== 调试代码结束 ==========================
``` 