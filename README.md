# EBR Fine-tune / Inference / Visualization 指南

本仓库已经将微调、批量推理与向量降维可视化集中在 `EBR_workflow.ipynb` Notebook 中。该文件整合了原来的 `ebr_finetune.py`、`ebr_infer.py` 与 `ebr_dim_reduction_vis.py`，可以按需开启/关闭不同阶段。

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

## 2. Notebook 使用流程

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

## 3. 数据约定

- 训练数据需要是 JSON 文件，包含 `anchor`/`positive` 字段（以及可选的 `negative`）。放置在 `FINETUNE_CONFIG.data_path` 下。  
- 推理数据默认使用 CSV，分隔符 `\001`，至少包含 `INFERENCE_CONFIG.description_column`（默认 `description`）。  
- 可视化阶段要求嵌入文件中存在 `vector` 字段（空格分隔的浮点数组）；如果有标签列（如 `label`），可以在 `VIS_CONFIG.label_column` 中指定用于着色。

## 4. 常见调整

- **Prompt**：`FINETUNE_CONFIG.prompt` 与 `INFERENCE_CONFIG.prompt` 可自定义任务提示词。  
- **LoRA 参数**：`lora_r / lora_alpha / lora_dropout` 控制 adapter 容量与正则。  
- **降维超参**：`tsne_perplexity`、`tsne_learning_rate`、`umap_neighbors` 可根据数据规模调整。  
- **输出列**：如果想保留更多字段到嵌入文件，可在 `INFERENCE_CONFIG.output_columns` 内列出所需列名。

## 5. 运行与调试建议

- 训练/推理前先用少量样本测试，确认字段命名与分隔符正确。  
- Notebook 会打印关键日志（如可训练参数数量、批处理进度、指标说明）。  
- 若使用中文标签，可在 `VIS_CONFIG.chinese_fonts` 中指定常用字体，确保 Matplotlib 正确显示。  
- 长时间训练建议配合 `tmux`、`screen` 或容器环境，避免会话断开。

如需扩展额外流程（例如评估脚本、Faiss 索引构建），可在 Notebook 末尾追加新的配置 + 函数单元并复用现有的配置模式即可。

