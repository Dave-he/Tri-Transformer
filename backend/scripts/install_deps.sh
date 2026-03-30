#!/usr/bin/env bash
set -euo pipefail

echo "=== Tri-Transformer 训练依赖安装 ==="

python3 -c "import torch" 2>/dev/null && echo "✅ torch 已安装，跳过" || {
    echo "📦 安装 PyTorch (CPU)..."
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
}

python3 -c "import modelscope" 2>/dev/null && echo "✅ modelscope 已安装，跳过" || {
    echo "📦 安装 modelscope..."
    pip install modelscope==1.18.0
}

python3 -c "import transformers" 2>/dev/null && echo "✅ transformers 已安装，跳过" || {
    echo "📦 安装 transformers..."
    pip install transformers==4.46.3
}

python3 -c "import datasets" 2>/dev/null && echo "✅ datasets 已安装，跳过" || {
    echo "📦 安装 datasets..."
    pip install datasets==3.1.0
}

python3 -c "import accelerate" 2>/dev/null && echo "✅ accelerate 已安装，跳过" || {
    echo "📦 安装 accelerate..."
    pip install accelerate==1.1.1
}

echo ""
echo "=== 验证安装 ==="
python3 -c "import torch, transformers; print('✅ torch', torch.__version__); print('✅ transformers', transformers.__version__)"
python3 -c "import modelscope; print('✅ modelscope', modelscope.__version__)" 2>/dev/null || echo "⚠️  modelscope 验证失败（可能版本不兼容）"

echo ""
echo "=== 安装完成 ==="
