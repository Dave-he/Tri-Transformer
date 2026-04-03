#!/bin/bash
# Tri-Transformer 项目自动化迭代脚本
# 用于 Linux/macOS 环境

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"
FRONTEND_DIR="$SCRIPT_DIR/../frontend"
EVAL_DIR="$SCRIPT_DIR/../eval"

write_header() {
    echo -e "\n============================================================"
    echo -e "$1"
    echo -e "============================================================\n"
}

install_dependencies() {
    write_header "安装后端依赖"
    cd "$BACKEND_DIR"
    pip install -r requirements.txt
    
    write_header "安装前端依赖"
    cd "$FRONTEND_DIR"
    npm install
    
    if [ -d "$EVAL_DIR" ]; then
        write_header "安装 Eval 依赖"
        cd "$EVAL_DIR"
        pip install -r requirements.txt
    fi
}

start_dev_servers() {
    write_header "启动开发服务器"
    
    # 启动后端
    cd "$BACKEND_DIR"
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    # 启动前端
    cd "$FRONTEND_DIR"
    pnpm dev &
    FRONTEND_PID=$!
    
    echo "后端运行在 http://localhost:8000 (PID: $BACKEND_PID)"
    echo "前端运行在 http://localhost:3000 (PID: $FRONTEND_PID)"
    echo "按 Ctrl+C 停止所有服务"
    
    wait
}

run_tests() {
    write_header "运行后端测试"
    cd "$BACKEND_DIR"
    python -m pytest tests/ -v --tb=short
    
    write_header "运行前端测试"
    cd "$FRONTEND_DIR"
    pnpm test
    
    if [ -d "$EVAL_DIR" ]; then
        write_header "运行 Eval 测试"
        cd "$EVAL_DIR"
        python -m pytest tests/ -v --tb=short
    fi
}

run_linters() {
    write_header "后端代码格式化"
    cd "$BACKEND_DIR"
    black app/ tests/
    flake8 app/ tests/
    
    write_header "前端代码检查"
    cd "$FRONTEND_DIR"
    pnpm lint
    pnpm typecheck
}

build_project() {
    write_header "构建前端"
    cd "$FRONTEND_DIR"
    pnpm build
    
    write_header "构建 Docker 容器"
    cd "$SCRIPT_DIR/.."
    docker-compose build
}

clean_project() {
    write_header "清理构建文件"
    
    # Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    
    # Node modules
    rm -rf "$FRONTEND_DIR/node_modules"
    
    # Build artifacts
    rm -rf "$FRONTEND_DIR/dist"
    
    echo "清理完成!"
}

# 主逻辑
case "${1:-all}" in
    install) install_dependencies ;;
    dev) start_dev_servers ;;
    test) run_tests ;;
    lint) run_linters ;;
    build) build_project ;;
    clean) clean_project ;;
    all)
        install_dependencies
        run_linters
        run_tests
        ;;
    *)
        echo "用法：$0 {install|dev|test|lint|build|clean|all}"
        exit 1
        ;;
esac

echo -e "\n操作完成!"
