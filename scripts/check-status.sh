#!/bin/bash
# Tri-Transformer 项目状态检查脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "\n========================================"
echo -e "Tri-Transformer 项目状态检查"
echo -e "========================================\n"

# 检查 Python 环境
echo -e "Python 环境:"
if command -v python &> /dev/null; then
    echo -e "  ✓ $(python --version)"
else
    echo -e "  ✗ Python 未安装或不在 PATH 中"
fi

# 检查 Node.js 环境
echo -e "\nNode.js 环境:"
if command -v node &> /dev/null; then
    echo -e "  ✓ Node.js $(node --version)"
else
    echo -e "  ✗ Node.js 未安装或不在 PATH 中"
fi

# 检查后端依赖
echo -e "\n后端依赖:"
if [ -d "$PROJECT_ROOT/backend/venv" ]; then
    echo -e "  ✓ 虚拟环境已创建"
else
    echo -e "  ⚠ 虚拟环境未创建"
fi

if [ -d "$PROJECT_ROOT/backend/__pycache__" ]; then
    echo -e "  ✓ 依赖已安装"
else
    echo -e "  ⚠ 可能需要安装依赖"
fi

# 检查前端依赖
echo -e "\n前端依赖:"
if [ -d "$PROJECT_ROOT/frontend/node_modules" ]; then
    echo -e "  ✓ node_modules 已安装"
else
    echo -e "  ⚠ 需要运行 npm install"
fi

# 检查 Docker
echo -e "\nDocker 环境:"
if command -v docker &> /dev/null; then
    echo -e "  ✓ $(docker --version)"
else
    echo -e "  ⚠ Docker 未安装 (可选)"
fi

# 检查环境变量
echo -e "\n环境变量配置:"
if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    echo -e "  ✓ backend/.env 已配置"
else
    echo -e "  ⚠ backend/.env 不存在，请复制 .env.example"
fi

if [ -f "$PROJECT_ROOT/frontend/.env" ]; then
    echo -e "  ✓ frontend/.env 已配置"
else
    echo -e "  ⚠ frontend/.env 不存在，请复制 .env.example"
fi

# 检查 Git
echo -e "\nGit 状态:"
if git rev-parse --git-dir > /dev/null 2>&1; then
    branch=$(git branch --show-current)
    echo -e "  ✓ 当前分支：$branch"
    
    if ! git diff --quiet; then
        echo -e "  ⚠ 有未提交的更改"
    else
        echo -e "  ✓ 工作区干净"
    fi
else
    echo -e "  ⚠ 不是 Git 仓库"
fi

# 检查关键文件
echo -e "\n关键文件:"
files=(
    "README.md"
    "LICENSE"
    "CONTRIBUTING.md"
    "docker-compose.yml"
    ".github/workflows/ci-cd.yml"
)

for file in "${files[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        echo -e "  ✓ $file"
    else
        echo -e "  ✗ $file 缺失"
    fi
done

# 检查测试
echo -e "\n测试状态:"
backend_tests=$(find "$PROJECT_ROOT/backend/tests" -name "test_*.py" 2>/dev/null | wc -l)
if [ "$backend_tests" -gt 0 ]; then
    echo -e "  ✓ 后端测试文件：$backend_tests 个"
else
    echo -e "  ⚠ 未找到后端测试文件"
fi

frontend_tests=$(find "$PROJECT_ROOT/frontend/src" -name "*.test.ts*" 2>/dev/null | wc -l)
if [ "$frontend_tests" -gt 0 ]; then
    echo -e "  ✓ 前端测试文件：$frontend_tests 个"
else
    echo -e "  ⚠ 未找到前端测试文件"
fi

echo -e "\n========================================"
echo -e "检查完成!"
echo -e "========================================\n"
