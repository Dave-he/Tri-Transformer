#!/usr/bin/env pwsh
# Tri-Transformer 项目自动化迭代脚本
# 用于 Windows PowerShell 环境

param(
    [ValidateSet('install', 'dev', 'test', 'lint', 'build', 'clean', 'all')]
    [string]$Action = 'all'
)

$ErrorActionPreference = "Stop"
$BackendDir = Join-Path $PSScriptRoot "backend"
$FrontendDir = Join-Path $PSScriptRoot "frontend"
$EvalDir = Join-Path $PSScriptRoot "eval"

function Write-Header {
    param([string]$Text)
    Write-Host "`n$('=' * 60)" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host $('=' * 60) -ForegroundColor Cyan
}

function Install-Dependencies {
    Write-Header "安装后端依赖"
    Set-Location $BackendDir
    pip install -r requirements.txt
    
    Write-Header "安装前端依赖"
    Set-Location $FrontendDir
    npm install
    
    Write-Header "安装 Eval 依赖"
    if (Test-Path $EvalDir) {
        Set-Location $EvalDir
        pip install -r requirements.txt
    }
}

function Start-DevServers {
    Write-Header "启动后端开发服务器"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$BackendDir'; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    
    Write-Header "启动前端开发服务器"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$FrontendDir'; pnpm dev"
    
    Write-Host "`n提示：后端运行在 http://localhost:8000" -ForegroundColor Green
    Write-Host "提示：前端运行在 http://localhost:3000" -ForegroundColor Green
}

function Run-Tests {
    Write-Header "运行后端测试"
    Set-Location $BackendDir
    python -m pytest tests/ -v --tb=short
    
    Write-Header "运行前端测试"
    Set-Location $FrontendDir
    pnpm test
    
    Write-Header "运行 Eval 测试"
    if (Test-Path $EvalDir) {
        Set-Location $EvalDir
        python -m pytest tests/ -v --tb=short
    }
}

function Run-Linters {
    Write-Header "后端代码格式化"
    Set-Location $BackendDir
    black app/ tests/
    flake8 app/ tests/
    
    Write-Header "前端代码检查"
    Set-Location $FrontendDir
    pnpm lint
    pnpm typecheck
}

function Build-Project {
    Write-Header "构建前端"
    Set-Location $FrontendDir
    pnpm build
    
    Write-Header "构建 Docker 容器"
    Set-Location $PSScriptRoot
    docker-compose build
}

function Clean-Project {
    Write-Header "清理构建文件"
    
    # Python cache
    Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
    
    # Node modules
    if (Test-Path "$FrontendDir\node_modules") {
        Remove-Item "$FrontendDir\node_modules" -Recurse -Force
    }
    
    # Build artifacts
    if (Test-Path "$FrontendDir\dist") {
        Remove-Item "$FrontendDir\dist" -Recurse -Force
    }
    
    Write-Host "清理完成!" -ForegroundColor Green
}

# 主逻辑
switch ($Action) {
    'install' { Install-Dependencies }
    'dev' { Start-DevServers }
    'test' { Run-Tests }
    'lint' { Run-Linters }
    'build' { Build-Project }
    'clean' { Clean-Project }
    'all' {
        Install-Dependencies
        Run-Linters
        Run-Tests
    }
}

Write-Host "`n操作完成!" -ForegroundColor Green
