#!/usr/bin/env python3
"""
ShopBot Reward Model 训练脚本 - 修复版
关键修复：分类层(score)需要单独训练，不在LoRA里
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import RewardTrainer, RewardConfig

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 配置区域 ============

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/reward_model")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/rlhf/preference_pairs_v1.jsonl")

LORA_RANK = 8
LORA_ALPHA = 32

BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_LENGTH = 512

# ==================================


def load_preference_data(data_path):
    """加载偏好对比数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })
    print(f"✅ 加载了 {len(data)} 对偏好数据")
    return data


def main():
    print("🚀 开始 ShopBot Reward Model 训练")
    print("=" * 50)
    print("💡 关键：LoRA只训练Transformer层，score层全量训练")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载tokenizer
    print("\n⏳ 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer加载完成")
    
    # 2. 加载基础模型
    print("\n⏳ 加载基础模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        torch_dtype=torch.float32,  # 用float32更稳定
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ 模型加载完成")
    
    # 3. 关键修复：只给transformer层加LoRA，score层保持可训练
    print("\n⏳ 配置LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 不包括score
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        # 关键：指定modules_to_save，这些模块会全量训练并保存
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    
    print("\n可训练参数统计:")
    model.print_trainable_parameters()
    
    # 验证score层确实可训练
    score_params = list(model.score.parameters())
    print(f"✅ score层参数数量: {sum(p.numel() for p in score_params):,}")
    print(f"✅ score层是否可训练: {score_params[0].requires_grad}")
    
    # 4. 加载数据
    print("\n⏳ 加载偏好数据...")
    raw_data = load_preference_data(DATA_PATH)
    dataset = Dataset.from_list(raw_data)
    
    print(f"\n📋 数据示例:")
    print(f"   Prompt: {dataset[0]['prompt'][:40]}...")
    print(f"   Chosen: {dataset[0]['chosen'][:40]}...")
    print(f"   Rejected: {dataset[0]['rejected'][:40]}...")
    
    # 5. 配置训练
    print("\n⏳ 配置训练...")
    reward_config = RewardConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        logging_steps=2,
        save_steps=10,
        bf16=False,  # 用float32更稳定
        fp16=False,
        report_to="none",
        max_length=MAX_LENGTH,
    )
    
    # 6. 创建RewardTrainer
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=reward_config,
    )
    print("✅ RewardTrainer创建完成")
    
    # 7. 开始训练
    print("\n🎬 开始训练 Reward Model！")
    print("📊 原理: chosen的分数要 > rejected的分数")
    trainer.train()
    
    # 8. 关键：保存完整模型（包括LoRA和全量训练的score层）
    print("\n💾 保存模型...")
    
    # 保存到临时目录
    temp_dir = os.path.join(OUTPUT_DIR, "temp_save")
    trainer.save_model(temp_dir)
    
    # 手动合并并保存（确保score层被正确保存）
    print("   合并LoRA权重...")
    merged_model = model.merge_and_unload()  # 合并LoRA到基础模型
    
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # 保存完整模型
    merged_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # 保存训练配置
    config_info = {
        "base_model": BASE_MODEL,
        "lora_rank": LORA_RANK,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "data_size": len(raw_data)
    }
    with open(os.path.join(final_dir, "rm_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"✅ Reward Model已保存到: {final_dir}")
    
    print("\n🎉 Reward Model训练完成！")


if __name__ == "__main__":
    main()
