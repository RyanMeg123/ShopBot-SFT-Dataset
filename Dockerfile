# ShopBot API Docker镜像
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型文件和代码
COPY outputs/ppo_model/final ./model
COPY api_server.py .

# 设置环境变量
ENV MODEL_PATH=/app/model
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "api_server.py"]
