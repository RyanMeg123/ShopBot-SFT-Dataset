#!/usr/bin/env python3
"""
ShopBot API 服务
基于 FastAPI 的模型推理服务
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

# ============ 配置 ============
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_PATH = os.environ.get("MODEL_PATH", "./outputs/ppo_model/final")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 256

# ============ 加载模型 ============
print("⏳ 正在加载模型...")
print(f"   设备: {DEVICE}")
print(f"   模型路径: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(DEVICE)

model = PeftModel.from_pretrained(model, MODEL_PATH)
model.eval()

print("✅ 模型加载完成！")

# ============ FastAPI 应用 ============
app = FastAPI(
    title="ShopBot API",
    description="电商客服AI助手推理服务",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 256


class ChatResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@app.get("/")
async def root():
    return {
        "message": "ShopBot API 服务运行中",
        "model": "ShopBot-PPO",
        "device": DEVICE
    }


@app.get("/health")
async def health():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口
    
    - **message**: 用户输入消息
    - **temperature**: 生成随机性 (0.0-1.0)
    - **max_tokens**: 最大生成token数
    """
    try:
        # 构建对话
        messages = [{"role": "user", "content": request.message}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True
        ).to(DEVICE)
        
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码输出
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        completion_tokens = outputs[0].shape[0] - prompt_tokens
        
        return ChatResponse(
            response=response,
            model="ShopBot-PPO",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口（SSE）
    """
    from fastapi.responses import StreamingResponse
    
    async def generate_stream():
        messages = [{"role": "user", "content": request.message}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # 流式生成
        with torch.no_grad():
            for i in range(request.max_tokens):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=request.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )
                
                new_token = outputs[0][-1:]
                text = tokenizer.decode(new_token, skip_special_tokens=True)
                
                if text:
                    yield f"data: {text}\n\n"
                
                inputs["input_ids"] = outputs
                
                # 检查是否结束
                if new_token.item() == tokenizer.eos_token_id:
                    break
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
