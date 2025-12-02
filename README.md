# EBR Fine-tune / Inference / Visualization 指南

## 1. 环境准备

1. 建议使用 Python 3.10+ 虚拟环境：  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖：  
   ```bash
   pip install -r requirements.txt
   ```
3. 如果需要 GPU 训练，确保本地 CUDA/cuDNN 版本与 `torch` 匹配。

## 2. 使用方式

### 2.1 Notebook 使用流程（推荐）

1. 启动 Notebook：  
   ```bash
   jupyter notebook EBR_workflow.ipynb
   ```
2. 阅读前两格 Markdown，确认流程说明。  
3. 在“配置”代码单元中按需修改：
   - `FINETUNE_CONFIG`: 训练数据路径、LoRA 参数、batch size 等。
   - `INFERENCE_CONFIG`: 推理模型路径、输入 CSV 目录、输出文件等。
   - `VIS_CONFIG`: 可视化所需的嵌入文件、标签列名、t-SNE/UMAP 参数等。
4. 设置底部的运行开关：
   ```python
   RUN_FINETUNE = True
   RUN_INFERENCE = True
   RUN_VISUALIZATION = True
   ```
   只执行需要的阶段，耗时/耗资源的步骤可保持 `False`。
5. 依次运行全部单元或只运行相关部分：
   - `RUN_FINETUNE` 为 True 时，会调用 `train_sentence_transformer` 完成 LoRA 微调并将模型保存到 `./saved_model/<model>-peft-lora/`。
   - `RUN_INFERENCE` 为 True 时，`run_batch_inference` 会遍历 `infer_data` 下的 CSV，按 `description_column` 取文本、批量编码并将结果写入 `./faiss/embeds_2.txt`（或你指定的位置）。
   - `RUN_VISUALIZATION` 为 True 时，`visualize_embeddings` 会读取嵌入文件，计算 t-SNE/UMAP 降维结果及聚类指标，并生成 `./embedding_scatter.png`。

### 2.2 命令行脚本使用

也可以直接使用 `ebr_finetune.py` 进行微调：

```bash
python ebr_finetune.py \
    --model_name ./model/KaLM-embedding-multilingual-mini-instruct-v2.5 \
    --data_path ./train_text \
    --output_dir ./saved_model \
    --noise_enabled \
    --noise_type random_delete \
    --noise_prob 0.1 \
    --num_epochs 1 \
    --train_batch_size 6 \
    --learning_rate 2e-4
```

**噪声参数说明**：
- `--noise_enabled`：启用文本噪声增强（防止过拟合）
- `--noise_type`：噪声类型，可选 `random_delete`（随机删除词）、`random_swap`（随机交换相邻词）、`char_delete`（随机删除字符）、`none`（禁用）
- `--noise_prob`：每个样本应用噪声的概率（0.0-1.0，建议 0.1-0.2）
- `--noise_apply_to_fields`：应用噪声的字段（默认：`anchor positive`）

## 3. 数据约定

- 训练数据需要是 JSON 文件，包含 `anchor`/`positive` 字段（以及可选的 `negative`）。放置在 `FINETUNE_CONFIG.data_path` 下。  
- 推理数据默认使用 CSV，分隔符 `\001`，至少包含 `INFERENCE_CONFIG.description_column`（默认 `description`）。  
- 可视化阶段要求嵌入文件中存在 `vector` 字段（空格分隔的浮点数组）；如果有标签列（如 `label`），可以在 `VIS_CONFIG.label_column` 中指定用于着色。

## 4. 常见调整

- **Prompt**：`FINETUNE_CONFIG.prompt` 与 `INFERENCE_CONFIG.prompt` 可自定义任务提示词。  
- **LoRA 参数**：`lora_r / lora_alpha / lora_dropout` 控制 adapter 容量与正则。  
- **降维超参**：`tsne_perplexity`、`tsne_learning_rate`、`umap_neighbors` 可根据数据规模调整。  
- **输出列**：如果想保留更多字段到嵌入文件，可在 `INFERENCE_CONFIG.output_columns` 内列出所需列名。
- **噪声增强（防过拟合）**：
  - 在 Notebook 中：设置 `FINETUNE_CONFIG.noise_enabled=True`，并调整 `noise_type`（可选：`"random_delete"`、`"random_swap"`、`"char_delete"`）和 `noise_prob`（0.0-1.0，建议 0.1-0.2）。
  - 在命令行脚本中：使用 `--noise_enabled --noise_type random_delete --noise_prob 0.1` 等参数。
  - 噪声仅应用于训练集，有助于防止模型过拟合单一数据模式。
- **正则化参数**：
  - `--weight_decay`: L2 正则化系数（默认 0.01，建议范围 0.01-0.1）
  - `--l1_regularization`: L1 正则化系数（默认 0.0，建议范围 1e-5 到 1e-3）
  - `--lora_dropout`: LoRA 层的 dropout 率（默认 0.1，建议范围 0.1-0.3）
  - `--max_grad_norm`: 梯度裁剪阈值（默认 1.0，0.0 禁用）
  - `--label_smoothing`: 标签平滑系数（默认 0.0，建议范围 0.0-0.1）
  - `--early_stopping_patience`: Early stopping 耐心值（默认 0 禁用，建议 3-5）
  - `--lr_scheduler_type`: 学习率调度器类型（默认 "linear"，可选: "cosine", "polynomial" 等）

## 5. 解决 GPU 内存溢出 (OOM) 问题

如果训练时遇到 `Device 0 OOM` 错误，可以尝试以下方法（按优先级排序）：

### 方法 1: 减小 batch size（最直接）
```bash
python ebr_finetune.py \
    --train_batch_size 2 \  # 从 6 减小到 2 或 1
    --eval_batch_size 1
```

### 方法 2: 使用梯度累积（推荐，保持有效 batch size）
```bash
python ebr_finetune.py \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4  # 有效 batch size = 2 * 4 = 8
```

### 方法 3: 启用梯度检查点（节省内存，训练稍慢）
```bash
python ebr_finetune.py \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing  # 启用梯度检查点
```

### 方法 4: 减小序列长度
```bash
python ebr_finetune.py \
    --max_seq_length 128  # 从 256 减小到 128
```

### 方法 5: 组合使用（最有效）
```bash
python ebr_finetune.py \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --max_seq_length 128 \
    --dataloader_num_workers 0 \
    --fp16  # 确保使用混合精度
```

### 内存优化参数说明
- `--gradient_accumulation_steps`: 梯度累积步数，增大可减少内存（有效 batch size = batch_size × accumulation_steps）
- `--gradient_checkpointing`: 启用梯度检查点，用计算时间换内存
- `--dataloader_num_workers`: 数据加载器工作进程数，设为 0 可减少内存占用
- `--dataloader_pin_memory`: 是否固定内存，OOM 时可禁用

## 6. 运行与调试建议

- 训练/推理前先用少量样本测试，确认字段命名与分隔符正确。  
- Notebook 会打印关键日志（如可训练参数数量、批处理进度、指标说明）。  
- 若使用中文标签，可在 `VIS_CONFIG.chinese_fonts` 中指定常用字体，确保 Matplotlib 正确显示。  
- 长时间训练建议配合 `tmux`、`screen` 或容器环境，避免会话断开。
- **OOM 排查顺序**：先减小 batch_size → 增加 gradient_accumulation_steps → 启用 gradient_checkpointing → 减小 max_seq_length

## 7. 训练异常诊断

### Loss 快速下降至接近 0

如果训练日志显示 loss 快速降至 0.0 或接近 0，可能的原因和解决方案：

**可能原因：**
1. **数据量过少**：训练集样本数 < 100，模型容易记住所有样本
2. **学习率过高**：导致模型快速过拟合
3. **数据重复**：训练数据中存在大量重复样本
4. **模型容量过大**：相对于数据量，模型参数过多

**解决方案：**
```bash
# 1. 降低学习率
python ebr_finetune.py --learning_rate 1e-4  # 从 2e-4 降低

# 2. 增加正则化（提高 dropout）
python ebr_finetune.py --lora_dropout 0.2  # 从 0.1 提高

# 3. 启用噪声增强（防止过拟合）
python ebr_finetune.py --noise_enabled --noise_prob 0.2

# 4. 增加权重衰减
python ebr_finetune.py --weight_decay 0.05  # 从 0.01 提高

# 5. 组合使用
python ebr_finetune.py \
    --learning_rate 1e-4 \
    --lora_dropout 0.2 \
    --weight_decay 0.05 \
    --noise_enabled \
    --noise_prob 0.2
```

**检查建议：**
- 查看验证集指标（eval loss），如果验证集 loss 不降反升，说明过拟合
- 检查训练集大小：`训练集大小: X` 日志
- 如果训练集 < 100 样本，建议增加数据量或使用更强的正则化

## 8. 正则化配置示例

### 基础正则化（防止过拟合）
```bash
python ebr_finetune.py \
    --weight_decay 0.05 \
    --lora_dropout 0.2 \
    --max_grad_norm 1.0
```

### 强正则化（数据量少时）
```bash
python ebr_finetune.py \
    --weight_decay 0.1 \
    --l1_regularization 1e-4 \
    --lora_dropout 0.3 \
    --max_grad_norm 0.5 \
    --noise_enabled \
    --noise_prob 0.2
```

### 使用 Early Stopping
```bash
python ebr_finetune.py \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.001 \
    --eval_steps 100  # 更频繁的评估
```

### 使用 Cosine 学习率调度
```bash
python ebr_finetune.py \
    --lr_scheduler_type cosine \
    --num_epochs 3
```

### 完整正则化配置示例
```bash
python ebr_finetune.py \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --l1_regularization 5e-5 \
    --lora_dropout 0.2 \
    --max_grad_norm 1.0 \
    --noise_enabled \
    --noise_prob 0.15 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.0001 \
    --lr_scheduler_type cosine \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4
```

### 正则化参数调优建议

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `weight_decay` | 0.01 | 0.01-0.1 | L2 正则化，越大越强 |
| `l1_regularization` | 0.0 | 1e-5 到 1e-3 | L1 正则化，促进稀疏性 |
| `lora_dropout` | 0.1 | 0.1-0.3 | Dropout 率，越大越强 |
| `max_grad_norm` | 1.0 | 0.5-2.0 | 梯度裁剪，防止梯度爆炸 |
| `label_smoothing` | 0.0 | 0.0-0.1 | 标签平滑（对 ranking loss 效果有限） |
| `early_stopping_patience` | 0 | 3-5 | Early stopping 耐心值 |
| `lr_scheduler_type` | linear | cosine/polynomial | 学习率调度策略 |

如需扩展额外流程（例如评估脚本、Faiss 索引构建），可在 Notebook 末尾追加新的配置 + 函数单元并复用现有的配置模式即可。

