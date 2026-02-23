#!/usr/bin/env python3
"""
ShopBot SFT 训练脚本
基于 Hugging Face TRL 库实现有监督微调
兼容 TRL >= 0.8.0
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 配置区域 ============

# 模型配置
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/sft_model")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/sft/train_v1.jsonl")

# LoRA配置
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# 训练配置
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 20  # 小数据集需要更多轮数
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

# ==================================


def load_data(data_path):
    """加载SFT数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    print(f"✅ 加载了 {len(data)} 条训练数据")
    return data


def format_conversation(example):
    """将对话格式化为模型输入"""
    conversation = example["conversation"]
    # 使用模型指定的chat template
    formatted = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted}


def main():
    global tokenizer
    
    print("🚀 开始 ShopBot SFT 训练")
    print(f"📦 基础模型: {BASE_MODEL}")
    print(f"📊 数据路径: {DATA_PATH}")
    print(f"💾 输出目录: {OUTPUT_DIR}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载tokenizer和模型
    print("\n⏳ 加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ 模型加载完成，参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 2. 配置LoRA
    print("\n⏳ 配置LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA配置完成")
    
    # 3. 加载数据
    print("\n⏳ 加载训练数据...")
    raw_data = load_data(DATA_PATH)
    
    # 转换为Hugging Face Dataset格式
    dataset = Dataset.from_list(raw_data)
    
    # 格式化数据
    dataset = dataset.map(format_conversation)
    print(f"✅ 数据格式化完成，示例：")
    print(f"   {dataset[0]['text'][:200]}...")
    
    # 4. 配置SFTConfig (新版TRL使用SFTConfig)
    print("\n⏳ 配置训练...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        logging_steps=5,
        save_steps=20,
        save_total_limit=2,
        bf16=True,  # Mac MPS 支持 bf16，不支持 fp16
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
    )
    
    # 5. 创建Trainer (新版API: args用SFTConfig, processing_class代替tokenizer)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    print("✅ Trainer创建完成")
    
    # 6. 开始训练
    print("\n🎬 开始训练！")
    trainer.train()
    
    # 7. 保存模型
    print("\n💾 保存模型...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"✅ 模型已保存到: {OUTPUT_DIR}/final")
    
    # 8. 保存训练配置
    config_info = {
        "base_model": BASE_MODEL,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "data_size": len(raw_data)
    }
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print("\n🎉 SFT训练完成！")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"📊 训练数据: {len(raw_data)} 条")
    print(f"🔄 训练轮数: {NUM_EPOCHS} 轮")


if __name__ == "__main__":
    main()
