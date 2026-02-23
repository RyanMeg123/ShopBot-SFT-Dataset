#!/bin/bash
# SFT训练一键启动脚本

echo "🚀 ShopBot SFT 训练启动器"
echo "=========================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"

# 检查依赖
echo ""
echo "📦 检查依赖..."
python3 -c "import torch, transformers, trl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  依赖未安装，正在安装..."
    pip install -r requirements.txt
fi
echo "✅ 依赖检查完成"

# 进入项目目录
cd "$(dirname "$0")/.."

# 运行训练
echo ""
echo "🎬 开始训练..."
echo "=========================="
python3 scripts/sft_train.py

echo ""
echo "=========================="
echo "训练完成！测试模型请运行:"
echo "  python3 scripts/test_model.py"
