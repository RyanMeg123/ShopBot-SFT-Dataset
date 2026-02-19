# 数据标注规范 v1.0

## 1. SFT数据标注规范

### 1.1 数据格式
```json
{
  "id": "sft_001",
  "category": "退换货", 
  "difficulty": "easy",
  "conversation": [
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "客服回答"}
  ],
  "metadata": {
    "product_type": "服装",
    "emotion": "neutral",
    "turns": 2
  }
}
```

### 1.2 场景分类
- 商品咨询
- 订单查询
- 退换货
- 物流跟踪
- 售后服务
- 优惠活动

### 1.3 质量要求
- ✅ 对话自然流畅
- ✅ 信息准确完整
- ✅ 符合客服礼仪
- ✅ 涵盖多轮对话
- ❌ 避免敏感信息
- ❌ 避免重复模板

## 2. RLHF偏好对标注规范

### 2.1 数据格式
```json
{
  "id": "rlhf_001",
  "prompt": "用户问题",
  "chosen": "优质回答",
  "rejected": "劣质回答",
  "category": "礼貌性",
  "reason": "chosen更礼貌且详细"
}
```

### 2.2 对比维度
- 准确性：信息是否正确
- 完整性：是否回答全面
- 礼貌性：语气和态度
- 简洁性：是否冗长
- 专业性：术语使用

### 2.3 标注原则
- chosen 必须明显优于 rejected
- 两者必须在同一维度对比
- 必须写明选择理由
