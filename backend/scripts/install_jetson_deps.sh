#!/bin/bash
# Tri-Transformer Jetson Nano 8GB 依赖安装脚本
# 适配 Jetson Nano (Maxwell GPU, CUDA 10.2, aarch64)
set -e

echo "=== Tri-Transformer Jetson Nano Setup ==="

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "⚠️  当前架构: $ARCH (非 aarch64), 此脚本专为 Jetson Nano 设计"
    echo "    继续? [y/N]"
    read -r CONFIRM
    if [ "$CONFIRM" != "y" ]; then
        exit 1
    fi
fi

if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  未检测到 /etc/nv_tegra_release, 可能非 Jetson 设备"
    echo "    继续? [y/N]"
    read -r CONFIRM
    if [ "$CONFIRM" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "1/6 系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3-pip python3-venv \
    build-essential cmake git \
    libopenblas-dev libffi-dev \
    libssl-dev curl wget

echo ""
echo "2/6 Python 虚拟环境..."
python3.10 -m venv /opt/tritransformer-venv || true
source /opt/tritransformer-venv/bin/activate
pip install --upgrade pip setuptools wheel

echo ""
echo "3/6 PyTorch (JetPack CUDA 10.2 wheel)..."
pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461

echo ""
echo "4/6 项目依赖..."
pip install --no-cache-dir \
    fastapi uvicorn sqlalchemy aiosqlite \
    pydantic pydantic-settings \
    python-jose[cryptography] passlib[bcrypt] \
    chromadb sentence-transformers \
    slowapi python-multipart \
    pyyaml

echo ""
echo "5/6 llama-cpp-python (CPU-only, 无 CUDA 加速)..."
CMAKE_ARGS="-DGGML_CUDA=off" pip install --no-cache-dir llama-cpp-python || {
    echo "⚠️  llama-cpp-python 安装失败, 尝试预编译 wheel..."
    pip install --no-cache-dir llama-cpp-python --prefer-binary
}

echo ""
echo "6/6 验证安装..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    total_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'GPU memory: {total_mem:.1f} GB')
" || echo "⚠️  PyTorch 验证失败"

python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" || echo "⚠️  FastAPI 验证失败"

python3 -c "import llama_cpp; print('llama-cpp-python: OK')" || echo "⚠️  llama-cpp-python 验证失败（可稍后安装）"

echo ""
echo "=== 安装完成 ==="
echo "激活环境: source /opt/tritransformer-venv/bin/activate"
echo "启动后端: cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "训练命令: python backend/scripts/train.py --jetson-nano --dataset dummy"
