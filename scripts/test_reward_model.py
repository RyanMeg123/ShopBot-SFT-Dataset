#!/usr/bin/env python3
"""
Reward Model 测试脚本 - 最终版
"""

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/reward_model/final")


def load_reward_model():
    """加载训练好的Reward Model"""
    print("⏳ 加载Reward Model...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        REWARD_MODEL_PATH, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH,
        num_labels=1,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()
    
    print("✅ Reward Model加载完成！")
    return model, tokenizer


def get_reward_score(model, tokenizer, prompt, response):
    """给单个回答打分"""
    text = f"{prompt}\n{response}"
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits[0].item()
    
    return score


def main():
    print("=" * 50)
    print("🤖 Reward Model 打分测试")
    print("=" * 50)
    
    model, tokenizer = load_reward_model()
    
    # 测试用例
    test_cases = [
        {
            "prompt": "你好，这件T恤有什么颜色？",
            "good": "您好！这款T恤目前有黑色、白色、灰色和藏青色四种颜色可选。请问您需要什么颜色呢？",
            "bad": "不知道，自己看网站。"
        },
        {
            "prompt": "我的订单什么时候到？",
            "good": "您好，我来帮您查询。订单已发货，预计明天下午送达。",
            "bad": "等着吧，到了就到了。"
        },
        {
            "prompt": "这个鞋子太大了，想退",
            "good": "好的，我们支持7天无理由退换货。请问您想退货还是换小一码？",
            "bad": "退不了，穿过了不能退。"
        }
    ]
    
    print("\n📊 测试结果:")
    print("=" * 50)
    
    all_correct = True
    for i, test in enumerate(test_cases, 1):
        score_good = get_reward_score(model, tokenizer, test["prompt"], test["good"])
        score_bad = get_reward_score(model, tokenizer, test["prompt"], test["bad"])
        
        print(f"\n【测试 {i}】{test['prompt'][:30]}...")
        print(f"  好回答: {score_good:+.2f}")
        print(f"  差回答: {score_bad:+.2f}")
        
        if score_good > score_bad:
            diff = score_good - score_bad
            print(f"  ✅ 正确 (好回答高 {diff:.2f} 分)")
        else:
            print(f"  ❌ 错误 (差回答更高)")
            all_correct = False
    
    print("\n" + "=" * 50)
    if all_correct:
        print("🎉 所有测试通过！Reward Model 工作正常")
    else:
        print("⚠️ 部分测试未通过，模型需要更多训练")
    print("=" * 50)
    
    # 交互测试
    print("\n💡 交互测试模式")
    print("   输入 prompt 和回答，查看 RM 分数")
    print("   输入 'quit' 退出")
    
    while True:
        print()
        prompt = input("📝 Prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        if not prompt:
            continue
            
        response = input("📝 回答: ").strip()
        if response.lower() == 'quit':
            break
        if not response:
            continue
        
        score = get_reward_score(model, tokenizer, prompt, response)
        print(f"   ⭐ RM 分数: {score:+.2f}")
        
        if score > 0:
            print("   👍 正面评价")
        elif score > -5:
            print("   😐 中性评价")
        else:
            print("   👎 负面评价")


if __name__ == "__main__":
    main()
