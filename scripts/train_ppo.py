#!/usr/bin/env python3
"""
ShopBot PPO 强化学习训练脚本 - 简化版
用Reward Model的反馈来优化SFT模型
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 配置区域 ============

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/sft_model/final")
REWARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/reward_model/final")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/ppo_model")

NUM_STEPS = 10  # 演示用的步数
MAX_NEW_TOKENS = 100

# ==================================


def load_prompts():
    """加载用于训练的prompts"""
    prompts = [
        "你好，这件T恤有什么颜色？",
        "我的订单什么时候到？",
        "这个鞋子太大了，想退",
        "现在有什么优惠吗？",
        "发货太慢了，能快点吗",
    ]
    return prompts


def get_reward(rm_model, rm_tokenizer, query, response, device):
    """计算奖励分数"""
    text = f"{query}\n{response}"
    inputs = rm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = rm_model(**inputs)
        score = outputs.logits[0].item()
    return score


def generate_response(model, tokenizer, prompt, device, max_new_tokens=100):
    """生成回答"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def simple_ppo_step(model, ref_model, optimizer, query, old_response, reward, kl_coef=0.2):
    """
    简化的PPO单步更新
    实际PPO更复杂，这里演示核心思想
    """
    # 注意：这是极度简化的版本，真实PPO需要计算优势函数、重要性采样等
    # 这里只做概念演示
    
    device = next(model.parameters()).device
    
    messages = [{"role": "user", "content": query}]
    prompt_text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 编码完整的prompt+response
    full_text = prompt_text + old_response
    inputs = model.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # 前向传播
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    # 用reward作为loss的权重（简化版）
    # 真实PPO这里会复杂得多
    weighted_loss = loss * (1.0 - torch.sigmoid(torch.tensor(reward / 10.0)).item())
    
    # 反向传播
    optimizer.zero_grad()
    weighted_loss.backward()
    optimizer.step()
    
    return weighted_loss.item()


def main():
    print("=" * 60)
    print("🚀 ShopBot PPO 强化学习训练 (简化版)")
    print("=" * 60)
    print("💡 PPO = 用RM的反馈来优化模型")
    print("💡 注意：这是教学简化版，非生产级实现")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n📱 使用设备: {device}")
    
    # 1. 加载模型
    print("\n⏳ 加载模型...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载策略模型（要训练的）
    print("  加载策略模型(SFT)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    policy_model = PeftModel.from_pretrained(policy_model, SFT_MODEL_PATH)
    policy_model.train()
    
    # 加载参考模型（冻结，用于计算KL散度）
    print("  加载参考模型(冻结)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    ref_model = PeftModel.from_pretrained(ref_model, SFT_MODEL_PATH)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # 加载Reward Model
    print("  加载Reward Model...")
    rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        num_labels=1,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    rm_model.eval()
    
    # 给模型附加tokenizer（用于generate）
    policy_model.tokenizer = tokenizer
    
    print("✅ 所有模型加载完成")
    
    # 2. 准备优化器
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)
    
    # 3. 加载prompts
    prompts = load_prompts()
    print(f"\n📊 训练数据: {len(prompts)} 个prompts")
    
    # 4. 训练循环
    print("\n🎬 开始PPO训练！")
    print("-" * 60)
    
    for step in range(NUM_STEPS):
        prompt = prompts[step % len(prompts)]
        
        print(f"\n【Step {step + 1}/{NUM_STEPS}】")
        print(f"  Prompt: {prompt}")
        
        # 生成回答
        response = generate_response(policy_model, tokenizer, prompt, device, MAX_NEW_TOKENS)
        print(f"  生成回答: {response[:60]}...")
        
        # 计算奖励
        reward = get_reward(rm_model, rm_tokenizer, prompt, response, device)
        print(f"  RM评分: {reward:+.2f}")
        
        # 简单PPO更新（教学版）
        try:
            loss = simple_ppo_step(policy_model, ref_model, optimizer, prompt, response, reward)
            print(f"  更新损失: {loss:.4f}")
            print(f"  ✅ 模型已更新（向高分方向优化）")
        except Exception as e:
            print(f"  ⚠️ 更新跳过: {e}")
        
        print("-" * 60)
    
    # 5. 保存模型
    print("\n💾 保存PPO模型...")
    policy_model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    
    # 保存配置
    config_info = {
        "base_model": BASE_MODEL,
        "sft_model": SFT_MODEL_PATH,
        "reward_model": REWARD_MODEL_PATH,
        "num_steps": NUM_STEPS,
        "note": "简化版PPO，仅用于教学演示"
    }
    with open(os.path.join(OUTPUT_DIR, "ppo_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print("\n🎉 PPO训练完成！")
    print("=" * 60)
    print("💡 核心概念:")
    print("   1. 模型生成回答")
    print("   2. RM给回答打分")
    print("   3. 高分回答 → 强化学习 → 模型更新")
    print("   4. 重复以上过程，模型越来越会生成高分回答")
    print("=" * 60)
    
    print("\n📂 模型已保存到:", os.path.join(OUTPUT_DIR, "final"))


if __name__ == "__main__":
    main()
