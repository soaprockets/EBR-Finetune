# -*- coding: utf-8 -*-
import os
import logging
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction
from sentence_transformers.losses import MultipleNegativesRankingLoss,MultipleNegativesSymmetricRankingLoss
from sentence_transformers.training_args import BatchSamplers
from peft import LoraConfig, TaskType
from transformers import TrainerCallback
import argparse

def generate_negative(dataset_Dict):
    candidate_pool = list(set(dataset_Dict["positive"]))
    negatives = []
    for pos in dataset_Dict["positive"]:
        while True:
            neg = random.choice(candidate_pool)
            if neg != pos:
                break
        negatives.append(neg)
    return negatives

def add_text_noise(text, noise_type="random_delete", noise_prob=0.1):
    """
    为文本添加噪声以防止过拟合
    
    Args:
        text: 输入文本
        noise_type: 噪声类型 ("random_delete", "random_swap", "char_delete", "none")
        noise_prob: 噪声应用概率
    
    Returns:
        添加噪声后的文本
    """
    if not text or noise_type == "none" or random.random() > noise_prob:
        return text
    
    if noise_type == "random_delete":
        # 随机删除词（保留语义）
        words = text.split()
        if len(words) <= 1:
            return text
        num_to_delete = max(1, int(len(words) * 0.05))  # 删除约5%的词
        indices_to_keep = sorted(random.sample(range(len(words)), len(words) - num_to_delete))
        return " ".join([words[i] for i in indices_to_keep])
    
    elif noise_type == "random_swap":
        # 随机交换相邻词
        words = text.split()
        if len(words) <= 1:
            return text
        for _ in range(int(len(words) * 0.02)):  # 交换约2%的词对
            if len(words) >= 2:
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)
    
    elif noise_type == "char_delete":
        # 随机删除字符（轻微扰动）
        chars = list(text)
        if len(chars) <= 2:
            return text
        num_to_delete = max(1, int(len(chars) * 0.01))  # 删除约1%的字符
        indices_to_keep = sorted(random.sample(range(len(chars)), len(chars) - num_to_delete))
        return "".join([chars[i] for i in indices_to_keep])
    
    return text

def apply_noise_to_example(example, noise_type="random_delete", noise_prob=0.1, apply_to_fields=None):
    """
    为数据集样本应用噪声增强
    
    Args:
        example: 数据集样本
        noise_type: 噪声类型
        noise_prob: 噪声应用概率
        apply_to_fields: 应用噪声的字段列表，None则默认["anchor", "positive"]
    
    Returns:
        添加噪声后的样本
    """
    if apply_to_fields is None:
        apply_to_fields = ["anchor", "positive"]
    
    for field in apply_to_fields:
        if field in example:
            if isinstance(example[field], list):
                example[field] = [
                    add_text_noise(str(item), noise_type, noise_prob) for item in example[field]
                ]
            elif isinstance(example[field], str):
                example[field] = add_text_noise(example[field], noise_type, noise_prob)
    
    return example

def add_prompt_to_text(text, prompt="Instruct: Retrieve semantically similar text.\nQuery:"):
    """在文本前添加 prompt"""
    if isinstance(text, str) and text.strip():
        return f"{prompt} {text}"
    return text

def add_prompt_to_example(example, prompt="Instruct: Retrieve semantically similar text.\nQuery:"):
    """为数据集样本添加 prompt
    
    Args:
        example: 数据集样本字典
        prompt: 要添加的 prompt 文本
    
    Returns:
        添加了 prompt 的样本
    """
    # 为所有文本字段添加 prompt
    if "anchor" in example:
        example["anchor"] = add_prompt_to_text(example["anchor"], prompt)
    if "positive" in example:
        example["positive"] = add_prompt_to_text(example["positive"], prompt)
    if "negative" in example:
        if isinstance(example["negative"], list):
            example["negative"] = [add_prompt_to_text(neg, prompt) for neg in example["negative"]]
        else:
            example["negative"] = add_prompt_to_text(example["negative"], prompt)
    # 处理其他可能的文本字段
    if "query" in example:
        example["query"] = add_prompt_to_text(example["query"], prompt)
    if "text" in example:
        example["text"] = add_prompt_to_text(example["text"], prompt)
    return example

def parse_args():
    parser = argparse.ArgumentParser(description="Sentence Transformer with LoRA Fine-tuning")
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default="./model/KaLM-embedding-multilingual-mini-instruct-v2.5",
                        help="Pretrained model name or path")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Whether to trust remote code when loading model")
    
    # LoRA配置
    parser.add_argument("--lora_r", type=int, default=64, #64 bge 微调默认值
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=128, #128 bge微调默认值
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    
    # Dense层配置
    parser.add_argument("--dense_dim1", type=int, default=512,
                        help="First dense layer dimension")
    parser.add_argument("--dense_dim2", type=int, default=256,
                        help="Second dense layer dimension")
    
    # 数据配置
    parser.add_argument("--data_path", type=str, default="./train_text/", # ./train_data bge模型最佳微调数据集； ./train_text qwen最佳微调数据集
                        help="Path to training data")
    parser.add_argument("--use_prompt", action="store_true", default=True,
                        help="Whether to add prompt to query texts")
    parser.add_argument("--prompt", type=str, default="Instruct: Retrieve semantically similar text.\nQuery:",
                        help="Prompt to prepend to query texts (only used when --use_prompt is set)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split ratio")
    parser.add_argument("--seed", type=int, default=12,
                        help="Random seed for data splitting")
    
    # 训练配置
    parser.add_argument("--output_dir", type=str, default="./saved_model/",
                        help="Output directory for saving models")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=6, #16
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, # 2e-4
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (L2 regularization coefficient)")
    parser.add_argument("--l1_regularization", type=float, default=0.0,
                        help="L1 regularization coefficient (0.0 to disable)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping (0.0 to disable)")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save steps")
    parser.add_argument("--warmup_steps", type=int, default=300,
                        help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging steps")
    
    # 正则化配置
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.0-1.0, 0.0 to disable)")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Early stopping patience (0 to disable, number of eval steps without improvement)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                        help="Early stopping threshold (minimum improvement to reset patience)")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    parser.add_argument("--lr_scheduler_kwargs", type=str, default=None,
                        help="Additional kwargs for lr scheduler (JSON string, e.g., '{\"num_cycles\": 2}')")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Whether to use fp16")
    parser.add_argument("--bf16", action="store_true", default=False),
    parser.add_argument("--max_seq_length", type=int, default=256)
    
    # 内存优化配置（解决 OOM 问题）
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (increase to reduce memory, e.g., 2, 4, 8)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to save memory (slower but uses less memory)")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of dataloader workers (0 for single process, reduce if OOM)")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=False,
                        help="Pin memory for dataloader (disable if OOM)")
    
    # 噪声/数据增强配置（防止过拟合）
    parser.add_argument("--noise_enabled", action="store_true", default=False,
                        help="Whether to enable text noise augmentation (default: False, use --noise_enabled to enable)")
    parser.add_argument("--noise_type", type=str, default="random_delete",
                        choices=["random_delete", "random_swap", "char_delete", "none"],
                        help="Type of noise to apply: random_delete, random_swap, char_delete, or none")
    parser.add_argument("--noise_prob", type=float, default=0.1,
                        help="Probability of applying noise to each sample (0.0-1.0)")
    parser.add_argument("--noise_apply_to_fields", type=str, nargs="+", default=["anchor", "positive"],
                        help="Fields to apply noise to (default: anchor positive)")
    
    return parser.parse_args()

def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params} || All params: {total_params} || Trainable %: {100 * trainable_params / total_params:.2f}")

class LossMonitorCallback(TrainerCallback):
    """监控训练 loss，检测异常情况"""
    def __init__(self):
        self.loss_history = []
        self.last_loss = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            current_loss = logs['loss']
            self.loss_history.append(current_loss)
            
            # 检测 loss 异常下降
            if self.last_loss is not None:
                loss_change = self.last_loss - current_loss
                loss_change_ratio = loss_change / self.last_loss if self.last_loss > 0 else 0
                
                # 如果 loss 下降超过 99% 或接近 0
                if current_loss < 0.001 and len(self.loss_history) > 5:
                    print(f'\n⚠️  警告: Loss 已降至 {current_loss:.6f}，可能出现过拟合或数据问题')
                    print(f'   建议检查: 1) 数据量是否足够 2) 学习率是否过高 3) 数据是否有重复')
                
                # 如果 loss 下降过快（单步下降超过 50%）
                if loss_change_ratio > 0.5 and self.last_loss > 0.1:
                    print(f'\n⚠️  注意: Loss 下降过快 ({self.last_loss:.4f} → {current_loss:.4f})')
                    print(f'   如果持续快速下降，可能需要降低学习率或增加正则化')
            
            self.last_loss = current_loss

class RegularizedLoss:
    """带 L1 正则化的 Loss 包装器"""
    def __init__(self, base_loss, model, l1_coef=0.0):
        self.base_loss = base_loss
        self.model = model
        self.l1_coef = l1_coef
    
    def __call__(self, sentence_features, labels):
        loss = self.base_loss(sentence_features, labels)
        
        # 添加 L1 正则化
        if self.l1_coef > 0:
            l1_reg = 0.0
            for param in self.model.parameters():
                if param.requires_grad:
                    l1_reg += torch.sum(torch.abs(param))
            loss = loss + self.l1_coef * l1_reg
        
        return loss
    
    def __getattr__(self, name):
        return getattr(self.base_loss, name)

class EarlyStoppingCallback(TrainerCallback):
    """Early Stopping 回调"""
    def __init__(self, patience=3, threshold=0.0, metric_name="eval_loss"):
        self.patience = patience
        self.threshold = threshold
        self.metric_name = metric_name
        self.best_metric = None
        self.patience_counter = 0
        
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs and self.metric_name in logs:
            current_metric = logs[self.metric_name]
            
            if self.best_metric is None:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                # 对于 loss，越小越好；对于 accuracy 等，越大越好
                if "loss" in self.metric_name.lower():
                    improvement = self.best_metric - current_metric
                else:
                    improvement = current_metric - self.best_metric
                
                if improvement > self.threshold:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    print(f'✓ 指标改善: {self.metric_name} = {current_metric:.6f} (改善 {improvement:.6f})')
                else:
                    self.patience_counter += 1
                    print(f'⚠️  指标未改善: {self.metric_name} = {current_metric:.6f} (patience: {self.patience_counter}/{self.patience})')
                    
                    if self.patience_counter >= self.patience:
                        print(f'\n🛑 Early Stopping: {self.patience} 次评估未改善，停止训练')
                        control.should_training_stop = True

def main():
    args = parse_args()
    
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    
    print(f'开始加载模型')
    base_model = SentenceTransformer(
        args.model_name,
        # truncate_dim=256
        # trust_remote_code=args.trust_remote_code
    )
    logging.info(base_model)
    
    # 应用 LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        # inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # target_modules=["Wo", "Wqkv"]
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj"]
    )
    base_model.add_adapter(lora_config)
    
    # 打印可训练参数
    print_trainable_parameters(base_model)
    
    # 内存优化提示
    if args.train_batch_size >= 8 or args.gradient_accumulation_steps == 1:
        print(f'内存优化提示: 当前 batch_size={args.train_batch_size}, gradient_accumulation_steps={args.gradient_accumulation_steps}')
        if args.train_batch_size >= 8:
            print('  - 如果遇到 OOM，建议减小 --train_batch_size (例如: 4, 2, 1)')
        if args.gradient_accumulation_steps == 1:
            print('  - 建议增加 --gradient_accumulation_steps (例如: 2, 4, 8) 来保持有效 batch size')
        if not args.gradient_checkpointing:
            print('  - 可以启用 --gradient_checkpointing 来节省内存（训练会稍慢）')
    
    # dense_layer1 = models.Dense(
    #     in_features=base_model.get_sentence_embedding_dimension(),
    #     out_features=args.dense_dim1,
    #     activation_function=nn.SiLU())
    # dense_layer2 = models.Dense(
        # in_features=base_model.get_sentence_embedding_dimension(),
        # out_features=args.dense_dim2,
        # activation_function=nn.GELU())
    # model = SentenceTransformer(modules=[base_model,dense_layer2])

    model = base_model
    
    model_name_only = args.model_name.split('/')[-1]
    print(f'模型加载完成')
    
    print(f'开始加载数据')
    json_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path)]
    dataset = load_dataset('json', data_files=json_files)
    dataset = dataset.filter(lambda example: example != '')
    
    # 根据参数决定是否添加 prompt 到所有文本字段
    if args.use_prompt:
        print(f'为数据添加 prompt: "{args.prompt}"')
        dataset = dataset.map(
            lambda x: add_prompt_to_example(x, args.prompt), 
            desc="Adding prompt to texts"
        )
        print(f'数据加载完成，已为所有查询文本添加 prompt')
    else:
        print(f'数据加载完成，未添加 prompt')
    
    dataset_dict = dataset['train'].train_test_split(test_size=args.test_size, seed=args.seed)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']
    
    # 打印数据集统计信息
    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}')
    if len(train_dataset) < 100:
        print(f'⚠️  警告: 训练集样本数较少 ({len(train_dataset)})，可能导致快速过拟合')
    if len(train_dataset) < 10:
        print(f'⚠️  严重警告: 训练集样本数过少 ({len(train_dataset)})，建议增加数据量')
    
    # 应用文本噪声增强（仅对训练集，防止过拟合）
    if args.noise_enabled:
        noise_fields = args.noise_apply_to_fields
        print(f'应用文本噪声增强: type={args.noise_type}, prob={args.noise_prob}, fields={noise_fields}')
        train_dataset = train_dataset.map(
            lambda x: apply_noise_to_example(x, args.noise_type, args.noise_prob, noise_fields),
            desc="Applying noise augmentation to training data"
        )
    
    # 启用梯度检查点以节省内存（如果启用）
    if args.gradient_checkpointing:
        if hasattr(base_model, 'enable_input_require_grads'):
            base_model.enable_input_require_grads()
        if hasattr(base_model[0], 'gradient_checkpointing_enable'):
            base_model[0].gradient_checkpointing_enable()
            print('已启用梯度检查点 (Gradient Checkpointing) 以节省内存')
    
    # 创建基础 loss
    base_loss = MultipleNegativesSymmetricRankingLoss(model)
    
    # 应用 L1 正则化（如果启用）
    # 创建基础 loss
    if args.l1_regularization > 0:
        # 定义一个工厂函数，接收 model 参数并返回 RegularizedLoss 实例
        def create_loss(model):
            base_loss = MultipleNegativesSymmetricRankingLoss(model)
            return RegularizedLoss(base_loss, model, args.l1_regularization)
        loss = create_loss  # 注意：这里传递的是函数，不是实例
        print(f'✓ 已启用 L1 正则化: coefficient = {args.l1_regularization}')
    else:
        loss = MultipleNegativesSymmetricRankingLoss  # 传递类本身，不是实例
    
    # 打印正则化配置
    print(f'\n正则化配置:')
    print(f'  - Weight Decay (L2): {args.weight_decay}')
    print(f'  - L1 Regularization: {args.l1_regularization if args.l1_regularization > 0 else "禁用"}')
    print(f'  - LoRA Dropout: {args.lora_dropout}')
    print(f'  - Max Grad Norm: {args.max_grad_norm if args.max_grad_norm > 0 else "禁用"}')
    print(f'  - Label Smoothing: {args.label_smoothing if args.label_smoothing > 0 else "禁用"}')
    if args.early_stopping_patience > 0:
        print(f'  - Early Stopping: patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}')
    print(f'  - LR Scheduler: {args.lr_scheduler_type}')
    
    run_name = f'{model_name_only}-peft-lora'
    
    # 解析 lr_scheduler_kwargs
    lr_scheduler_kwargs = {}
    if args.lr_scheduler_kwargs:
        import json
        try:
            lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)
        except:
            print(f'⚠️  警告: 无法解析 lr_scheduler_kwargs，使用默认值')
    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(args.output_dir, run_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=3,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, run_name),
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        lr_scheduler_type=args.lr_scheduler_type,
        **lr_scheduler_kwargs
    )
    
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=generate_negative(eval_dataset),
        main_similarity_function=SimilarityFunction.COSINE,
        name="sts-dev"
    )
    
    print(f'开始训练')
    # 添加训练监控回调
    callbacks = [LossMonitorCallback()]
    
    # 添加 Early Stopping 回调（如果启用）
    if args.early_stopping_patience > 0:
        early_stopping = EarlyStoppingCallback(
            patience=args.early_stopping_patience,
            threshold=args.early_stopping_threshold,
            metric_name="eval_loss"
        )
        callbacks.append(early_stopping)
        print(f'✓ 已启用 Early Stopping: patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}')
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=[dev_evaluator]
        # max_seq_length=args.max_seq_length
    )
    
    # 添加回调（如果支持）
    for callback in callbacks:
        try:
            trainer.add_callback(callback)
        except:
            # 如果不支持 add_callback，尝试通过 callbacks 参数
            try:
                trainer.callback_handler.add_callback(callback)
            except:
                print(f'注意: 无法添加回调 {type(callback).__name__}，但训练会继续')
    
    trainer.train()
    print(f'训练完成')
    
    print(f'开始保存模型')
    final_output_dir = os.path.join(args.output_dir, run_name)
    model.save_pretrained(final_output_dir)
    print(f'模型保存完成')

if __name__ == "__main__":
    main()