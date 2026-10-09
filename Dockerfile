FROM python:3.13-slim

WORKDIR /app

# 依赖锁定安装，避免 numpy 版本漂移
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认：运行端到端演示（落盘 benchmark.json）
CMD ["python", "-m", "examples.run_demo"]
