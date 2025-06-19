# 终极方案 (Mode 3) 实现路线图

这是一个结合了详细**实现策略**和清晰**路线图**的完整计划。两者将一一对应，确保每一步都清晰明确。

---

### **实现策略：代码修改与添加描述**

这部分详细说明了需要在哪些文件的哪个部分进行何种修改或添加。

#### **1. 准备工作：在 `ComposeTools.py` 中添加Mode 3框架**

这是构建方案骨架的准备步骤，完全遵循Mode 1和Mode 2的现有结构。

*   **1.1. 添加Mode 3专用配置**:
    *   **位置**: `ComposeTools.py` -> `ComposeConfig` 类。
    *   **操作**: 在该类中，添加终极方案所需的所有超参数，例如：
        *   `mode3_pretraining_epochs: int`
        *   `mode3_finetuning_epochs: int`
        *   `mode3_lambda_data: float` (总数据损失权重)
        *   `mode3_lambda_pde_initial: float` (预训练阶段的PDE损失权重，应为0)
        *   `mode3_lambda_pde_final: float` (微调阶段的PDE损失权重)
        *   `mode3_lambda_pseudo_initial: float` (预训练阶段伪数据损失的强度调节因子)
        *   `mode3_lambda_pseudo_final: float` (微调阶段伪数据损失的强度调节因子)
        *   `mode3_pseudo_grid_density: int` (伪数据网格点的总数)
        *   `mode3_variance_epsilon: float` (权重分母防零的小数)

*   **1.2. 添加Mode 3专用类**:
    *   **位置**: `ComposeTools.py`，与 `Mode1ResidualKriging`, `Mode2ROIDetector` 等类并列。
    *   **操作**:
        *   **创建 `Mode3KrigingGuided` 类**: 这个类将负责您方案中的**第一步**。它将包含两个主要方法：`build_kriging_surrogate` 和 `generate_pseudo_data`。
        *   **创建 `Mode3PINNTrainer` 类**: 这个类将负责您方案中的**第三步**。它将包含一个核心方法 `staged_training`，用于编排预训练和物理微调两个阶段。

*   **1.3. 在工作流中注册Mode 3**:
    *   **位置**: `ComposeTools.py` -> `CouplingWorkflow` 类。
    *   **操作**:
        *   在 `__init__` 方法中，为Mode 3初始化一个工具集入口，例如 `self.mode3_tools = {}` (具体实现将在 `run_mode3_pipeline` 中完成)。
        *   创建一个新的工作流方法 `run_mode3_pipeline`。这个方法的结构将与 `run_mode1_pipeline` 和 `run_mode2_pipeline` 类似。

#### **2. 实现Kriging代理与伪数据生成（对应用户方案第一步）**

*   **位置**: `ComposeTools.py` -> 新创建的 `Mode3KrigingGuided` 类中。
*   **操作**:
    *   **实现 `build_kriging_surrogate` 方法**:
        *   此方法接收稀疏的真实数据点 (`train_points`, `train_values`)。
        *   它将利用现有的 `KrigingAdapter` 来训练一个Kriging模型，并将训练好的模型存储在类的实例变量中（例如 `self.kriging_surrogate`）。
    *   **实现 `generate_pseudo_data` 方法**:
        *   此方法不接收参数，但会使用已训练好的 `self.kriging_surrogate`。
        *   它首先根据 `config.mode3_pseudo_grid_density` 在计算域内生成一个规则的网格点 `x_pseudo`。
        *   然后，调用 `self.kriging_surrogate.predict(x_pseudo, return_std=True)` 来获取伪数据点的预测值 `y_pseudo` 和对应的标准差（或方差）`var_pseudo`。
        *   **核心**：根据您提供的公式 `w(x) = 1 / (var_kriging(x) + ε)` 计算每个伪数据点的权重 `w_pseudo`。
        *   最后，返回一个包含 `{'points': x_pseudo, 'values': y_pseudo, 'weights': w_pseudo}` 的字典。

#### **3. 为PINN核心启用加权损失功能（对应用户方案第二步）**

这是技术上最关键的一步，它需要修改底层的PINN封装。

*   **位置**: `PINN/pinn_core.py` -> `PINNTrainer` 类。
*   **操作**:
    *   **修改 `create_pinn_model` 方法**:
        *   为该方法添加一个新的可选参数 `sample_weights: Optional[np.ndarray] = None`。
        *   在方法内部，增加一个条件判断：`if sample_weights is not None:`。
    *   **在条件判断为True时**:
        *   **数据打包**: 创建一个 `(N, 2)` 的新数组，其中第一列是真实的对数剂量 `sampled_log_doses_values`，第二列是 `sample_weights`。
        *   **自定义损失函数**: 在 `PINNTrainer` 类中定义一个新的内部方法，例如 `def weighted_mse_loss(self, y_true_and_weights, y_pred):`。
            *   在此函数内部，需要从 `y_true_and_weights` 中解包出真实值 `y_true` 和权重 `weights`。
            *   然后计算并返回加权均方误差：`dde.backend.mean(weights * (y_pred - y_true) ** 2)`。
        *   **绑定数据和损失**: 在创建 `dde.data.PointSetBC` 时，将**打包后**的数据作为 `y_train` 参数传入。在调用 `self.model.compile()` 时，将 `loss_fn` 参数设置为您刚创建的 `self.weighted_mse_loss`。
    *   **在条件判断为False时 (即 `sample_weights` is None)**:
        *   保持现有的逻辑不变，使用标准的MSE损失。

#### **4. 编排分阶段训练流程（对应用户方案第三步）**

*   **位置**: `ComposeTools.py` -> 新创建的 `Mode3PINNTrainer` 类中的 `staged_training` 方法。
*   **操作**:
    *   **实例化 `PINNAdapter`**。
    *   **阶段一：预训练**:
        *   **数据准备**: 将真实的 `train_points`, `train_values` 与 `pseudo_data['points']`, `pseudo_data['values']` 合并。
        *   **权重准备**: 构造一个总的权重数组。真实数据点的权重为1.0，伪数据点的权重为 `pseudo_data['weights']` 乘以 `config.mode3_lambda_pseudo_initial`。
        *   **调用训练**: 调用 `pinn_adapter.fit_from_memory`，传入合并后的数据、合并后的权重，并将 `loss_weights` 参数设置为 `[config.mode3_lambda_data, config.mode3_lambda_pde_initial]` (后者应为0)。
    *   **阶段二：物理微调**:
        *   **准备新参数**: 确定微调阶段的 `loss_weights`，即 `[config.mode3_lambda_data, config.mode3_lambda_pde_final]`。
        *   **调用继续训练**: 为了在不销毁模型的前提下继续训练，您需要一个能更新`loss_weights`并继续训练的机制。在 `PINNAdapter` 中创建一个新方法 `retrain_with_new_weights(self, epochs, loss_weights)`。
            *   这个新方法内部需要调用 `PINNTrainer` 中一个对应的方法（例如 `recompile_model`），该方法只做一件事：调用 `self.model.compile()` 以应用新的 `loss_weights`。
            *   然后，`retrain_with_new_weights` 再调用 `self.trainer.train()` 继续训练。
        *   **注意**: 在这个方案中，样本权重在模型创建时已绑定，微调阶段我们不更换样本权重，只调整 `loss_weights`（特别是PDE项的权重）。

#### **5. 在 `main.py` 中集成并提供命令行接口**

*   **位置**: `ComposeProject/main.py`
*   **操作**:
    *   在 `main` 函数的模式选择部分，添加 `elif args.mode == 3:` 的分支。
    *   在这个分支中，从 `args` 中读取Mode 3的配置，更新 `config` 对象。
    *   调用 `workflow.run_mode3_pipeline(...)`。
    *   在 `argparse` 部分，为Mode 3在 `ComposeConfig` 中添加的所有新参数（如 `--mode3-pretraining-epochs` 等）添加对应的命令行参数，并提供帮助说明。

---

### **实现路线图**

这是一个高层次的、分步执行的路线图，与上面的策略一一对应。

*   **路线图第一步：构建Mode 3框架**
    *   **任务**: 在 `ComposeTools.py` 中添加 `ComposeConfig` 的新参数、空的 `Mode3KrigingGuided` 和 `Mode3PINNTrainer` 类，并在 `CouplingWorkflow` 中添加 `run_mode3_pipeline` 的空方法。同时，在 `main.py` 中添加模式3的命令行参数和调用分支。
    *   **产出**: 一个可以被 `--mode 3` 参数调用、但不会执行任何操作的框架。

*   **路线图第二步：实现Kriging代理与伪数据生成**
    *   **任务**: 填充 `Mode3KrigingGuided` 类中的 `build_kriging_surrogate` 和 `generate_pseudo_data` 方法。
    *   **产出**: 一个能够根据稀疏数据训练Kriging模型，并生成带权重的伪数据点的功能模块。

*   **路线图第三步：为PINN核心启用加权损失功能**
    *   **任务**: **集中精力修改 `PINN/pinn_core.py`**。按照策略描述，修改 `PINNTrainer.create_pinn_model` 以支持 `sample_weights`，包括数据打包和实现自定义的 `weighted_mse_loss`。
    *   **产出**: 一个功能增强的 `PINNTrainer`，能够理解并利用样本权重进行训练。**这是整个方案的技术核心**。

*   **路线图第四步：编排分阶段训练流程**
    *   **任务**: 填充 `Mode3PINNTrainer.staged_training` 方法。实现预训练和物理微调两个阶段的逻辑，包括数据的合并、权重的构造，以及对 `PINNAdapter` 的两次调用（一次 `fit_from_memory`，一次 `retrain_with_new_weights`）。同时，在 `PINNAdapter` 中实现 `retrain_with_new_weights` 方法。
    *   **产出**: 一个完整的、自动化的分阶段训练控制器。

*   **路线图第五步：联调与测试**
    *   **任务**: 将所有部分连接起来。在 `run_mode3_pipeline` 中，依次调用 `Mode3KrigingGuided` 和 `Mode3PINNTrainer` 的方法。然后，通过命令行运行 `main.py --mode 3`，并调整超参数，观察其性能是否优于基线PINN。
    *   **产出**: 一个功能完整、可运行、可评估的"终极方案"实现。 