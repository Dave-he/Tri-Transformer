#!/bin/bash
# 在 Jetson Nano 上构建 llama.cpp (aarch64, 无 CUDA 加速)
set -e

echo "=== 构建 llama.cpp for Jetson Nano ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../llama.cpp-build"
INSTALL_DIR="${SCRIPT_DIR}/../llama.cpp-install"

if [ -d "${BUILD_DIR}" ]; then
    echo "已有 llama.cpp 源码, 拉取最新..."
    cd "${BUILD_DIR}" && git pull || true
else
    echo "克隆 llama.cpp 源码..."
    git clone https://github.com/ggerganov/llama.cpp.git "${BUILD_DIR}"
fi

cd "${BUILD_DIR}"

echo "编译 (CPU-only, aarch64)..."
cmake -B build \
    -DGGML_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
cmake --build build --config Release -j$(nproc)
cmake --install build

echo ""
echo "=== 构建完成 ==="
echo "安装目录: ${INSTALL_DIR}"
echo "量化工具: ${INSTALL_DIR}/bin/llama-quantize"
echo ""
echo "用法:"
echo "  ${INSTALL_DIR}/bin/llama-quantize model-F16.gguf Q5_K_M"
