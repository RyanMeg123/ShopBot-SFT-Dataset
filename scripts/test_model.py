#!/usr/bin/env python3
"""
ShopBot SFT 模型测试脚本
测试微调后的模型效果
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

# 模型路径
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs/sft_model/final")


def load_model():
    """加载微调后的模型"""
    print("⏳ 加载模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()  # 合并权重加速推理
    
    print("✅ 模型加载完成！")
    return model, tokenizer


def chat(model, tokenizer, user_input):
    """与模型对话"""
    messages = [
        {"role": "system", "content": "你是一个专业的电商客服助手，热情、耐心地回答用户问题。"},
        {"role": "user", "content": user_input}
    ]
    
    # 应用chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,  # 降低随机性，更接近训练数据
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手回复（去掉prompt部分）
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response


def main():
    print("=" * 50)
    print("🤖 ShopBot 客服助手测试")
    print("=" * 50)
    
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("提示：请先运行 sft_train.py 完成训练")
        sys.exit(1)
    
    # 预设测试用例
    test_cases = [
        "你好，这件T恤有什么颜色？",
        "我的订单什么时候到？",
        "这个鞋子太大了，想退",
        "现在有什么优惠吗？",
    ]
    
    print("\n📋 预设测试用例：")
    for i, test in enumerate(test_cases, 1):
        print(f"  {i}. {test}")
    
    print("\n💡 输入数字(1-4)选择测试用例，或直接输入问题")
    print("💡 输入 'quit' 退出")
    print("-" * 50)
    
    while True:
        user_input = input("\n📝 输入: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 再见！")
            break
        
        # 处理数字选择
        if user_input.isdigit() and 1 <= int(user_input) <= len(test_cases):
            user_input = test_cases[int(user_input) - 1]
            print(f"📝 问题: {user_input}")
        
        print("⏳ 生成回复...")
        response = chat(model, tokenizer, user_input)
        print(f"🤖 回复: {response}")


if __name__ == "__main__":
    main()
