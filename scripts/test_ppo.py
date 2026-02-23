#!/usr/bin/env python3
"""
ShopBot PPO 模型测试脚本
对比 SFT模型 vs PPO模型 的效果
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型路径
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/sft_model/final")
PPO_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/ppo_model/final")
REWARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/reward_model/final")


def load_model_for_comparison(model_path, model_name, device):
    """加载模型"""
    print(f"\n⏳ 加载 {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    print(f"✅ {model_name} 加载完成")
    return model, tokenizer


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
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def get_reward_score(rm_model, rm_tokenizer, prompt, response, device):
    """RM打分"""
    text = f"{prompt}\n{response}"
    inputs = rm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = rm_model(**inputs)
        score = outputs.logits[0].item()
    
    return score


def compare_models(sft_model, sft_tokenizer, ppo_model, ppo_tokenizer, 
                   rm_model, rm_tokenizer, prompt, device):
    """对比两个模型的回答"""
    
    # SFT生成
    sft_response = generate_response(sft_model, sft_tokenizer, prompt, device)
    sft_score = get_reward_score(rm_model, rm_tokenizer, prompt, sft_response, device)
    
    # PPO生成
    ppo_response = generate_response(ppo_model, ppo_tokenizer, prompt, device)
    ppo_score = get_reward_score(rm_model, rm_tokenizer, prompt, ppo_response, device)
    
    return {
        "prompt": prompt,
        "sft_response": sft_response,
        "sft_score": sft_score,
        "ppo_response": ppo_response,
        "ppo_score": ppo_score
    }


def print_comparison(result, idx):
    """打印对比结果"""
    print(f"\n{'='*60}")
    print(f"【测试 {idx}】{result['prompt']}")
    print(f"{'='*60}")
    
    print(f"\n📝 SFT模型回答:")
    print(f"   {result['sft_response'][:100]}{'...' if len(result['sft_response']) > 100 else ''}")
    print(f"   ⭐ RM分数: {result['sft_score']:+.2f}")
    
    print(f"\n📝 PPO模型回答:")
    print(f"   {result['ppo_response'][:100]}{'...' if len(result['ppo_response']) > 100 else ''}")
    print(f"   ⭐ RM分数: {result['ppo_score']:+.2f}")
    
    # 判断哪个更好
    if result['ppo_score'] > result['sft_score']:
        diff = result['ppo_score'] - result['sft_score']
        print(f"\n🏆 结果: PPO更优 (高 {diff:.2f} 分)")
    elif result['sft_score'] > result['ppo_score']:
        diff = result['sft_score'] - result['ppo_score']
        print(f"\n🏆 结果: SFT更优 (高 {diff:.2f} 分)")
    else:
        print(f"\n⚖️ 结果: 两者相当")


def main():
    print("="*60)
    print("🤖 ShopBot 模型对比测试")
    print("="*60)
    print("对比: SFT模型 vs PPO模型")
    print("="*60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n📱 使用设备: {device}")
    
    # 加载模型
    try:
        sft_model, sft_tokenizer = load_model_for_comparison(SFT_MODEL_PATH, "SFT模型", device)
        ppo_model, ppo_tokenizer = load_model_for_comparison(PPO_MODEL_PATH, "PPO模型", device)
        
        # 加载RM
        print("\n⏳ 加载Reward Model...")
        rm_tokenizer = AutoTokenizer.from_pretrained(
            REWARD_MODEL_PATH, 
            trust_remote_code=True,
            local_files_only=True
        )
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_PATH,
            num_labels=1,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
        rm_model.eval()
        print("✅ Reward Model 加载完成")
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("请确认以下模型已训练:")
        print(f"  - {SFT_MODEL_PATH}")
        print(f"  - {PPO_MODEL_PATH}")
        print(f"  - {REWARD_MODEL_PATH}")
        return
    
    # 测试用例
    test_prompts = [
        "你好，这件T恤有什么颜色？",
        "我的订单什么时候到？",
        "这个鞋子太大了，想退",
        "现在有什么优惠吗？",
        "发货太慢了，能快点吗",
        "这件衣服颜色太深了",
        "怎么取消订单？",
        "退货的钱多久到账？",
    ]
    
    print("\n" + "="*60)
    print("📊 开始对比测试")
    print("="*60)
    
    results = []
    for i, prompt in enumerate(test_prompts, 1):
        result = compare_models(
            sft_model, sft_tokenizer,
            ppo_model, ppo_tokenizer,
            rm_model, rm_tokenizer,
            prompt, device
        )
        results.append(result)
        print_comparison(result, i)
    
    # 统计总结
    print("\n" + "="*60)
    print("📈 测试总结")
    print("="*60)
    
    ppo_wins = sum(1 for r in results if r['ppo_score'] > r['sft_score'])
    sft_wins = sum(1 for r in results if r['sft_score'] > r['ppo_score'])
    ties = len(results) - ppo_wins - sft_wins
    
    print(f"\n总测试数: {len(results)}")
    print(f"PPO胜出: {ppo_wins} 次")
    print(f"SFT胜出: {sft_wins} 次")
    print(f"平局: {ties} 次")
    
    avg_sft = sum(r['sft_score'] for r in results) / len(results)
    avg_ppo = sum(r['ppo_score'] for r in results) / len(results)
    
    print(f"\n平均分数:")
    print(f"  SFT: {avg_sft:+.2f}")
    print(f"  PPO: {avg_ppo:+.2f}")
    
    if avg_ppo > avg_sft:
        print(f"\n🎉 结论: PPO模型平均表现更优 (高 {avg_ppo - avg_sft:.2f} 分)")
    else:
        print(f"\n📝 结论: SFT模型平均表现更优 (高 {avg_sft - avg_ppo:.2f} 分)")
    
    # 交互测试
    print("\n" + "="*60)
    print("💡 交互测试模式")
    print("输入你自己的prompt，对比两个模型的回答")
    print("输入 'quit' 退出")
    print("="*60)
    
    while True:
        print()
        prompt = input("📝 Prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        if not prompt:
            continue
        
        print("\n⏳ 生成中...")
        result = compare_models(
            sft_model, sft_tokenizer,
            ppo_model, ppo_tokenizer,
            rm_model, rm_tokenizer,
            prompt, device
        )
        print_comparison(result, "交互")


if __name__ == "__main__":
    main()
