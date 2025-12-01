import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer


######################################### 模型结构 ####################################################

data = pd.concat([
    pd.read_csv(file_path, sep='\001', names=[],#输入列
     engine="python") 
    for file_path in Path("./infer_data").iterdir()])
data = data[[]].drop_duplicates()
print(f'数据维度：{data.shape}')

model = SentenceTransformer("./saved_model/KaLM-embedding-multilingual-mini-instruct-v2.5-peft-lora/checkpoint-500",truncate_dim=256)


batch_size=512

# 6. 模型推理
with open("./faiss/embeds_2.txt", 'w') as f:
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        
        # 模型处理
        batch_data = data[start_idx:end_idx]

        # descriptions = [f"Instruct:给定一段房源信息描述，查询其他语义相似度高的房源描述，尤其是描述中的ID值。\nQuery {desc}" for desc in batch_data['description'].values.tolist()]
        # embeds = model.encode(descriptions, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
        embeds = model.encode(batch_data['description'].values.tolist(), normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True
        ,prompt="Instruct: Retrieve semantically similar text.\nQuery:"
        ) # 任务提示词 qwen系列模型
        # embeds = get_embeddings(batch_data['description'].values.tolist())
        result = [' '.join(str(x) for x in embed) for embed in embeds]
        
        # 数据融合
        outcome = pd.concat([batch_data.reset_index(drop=True), pd.DataFrame(result, columns=["vecs"])], axis=1)
        outcome = outcome[[]] # 需要返回的列
        
        # 写入本地文件
        outcome.to_csv(f, index=False, header=False, sep="\001", mode='a')
        print(f"Batch {start_idx // batch_size + 1} written to file")
    
print("模型处理 and 数据融合 is OK")
