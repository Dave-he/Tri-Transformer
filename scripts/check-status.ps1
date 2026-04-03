#!/usr/bin/env pwsh
# Tri-Transformer 项目状态检查脚本

param(
    [switch]$Detailed
)

$ErrorActionPreference = "SilentlyContinue"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Tri-Transformer 项目状态检查" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 检查 Python 环境
Write-Host "Python 环境:" -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion) {
    Write-Host "  ✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python 未安装或不在 PATH 中" -ForegroundColor Red
}

# 检查 Node.js 环境
Write-Host "`nNode.js 环境:" -ForegroundColor Yellow
$nodeVersion = node --version 2>&1
if ($nodeVersion) {
    Write-Host "  ✓ Node.js $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Node.js 未安装或不在 PATH 中" -ForegroundColor Red
}

# 检查后端依赖
Write-Host "`n后端依赖:" -ForegroundColor Yellow
$backendDir = Join-Path $PSScriptRoot "backend"
if (Test-Path "$backendDir\venv") {
    Write-Host "  ✓ 虚拟环境已创建" -ForegroundColor Green
} else {
    Write-Host "  ⚠ 虚拟环境未创建" -ForegroundColor Yellow
}

if (Test-Path "$backendDir\__pycache__") {
    Write-Host "  ✓ 依赖已安装" -ForegroundColor Green
} else {
    Write-Host "  ⚠ 可能需要安装依赖" -ForegroundColor Yellow
}

# 检查前端依赖
Write-Host "`n前端依赖:" -ForegroundColor Yellow
$frontendDir = Join-Path $PSScriptRoot "frontend"
if (Test-Path "$frontendDir\node_modules") {
    Write-Host "  ✓ node_modules 已安装" -ForegroundColor Green
} else {
    Write-Host "  ⚠ 需要运行 npm install" -ForegroundColor Yellow
}

# 检查 Docker
Write-Host "`nDocker 环境:" -ForegroundColor Yellow
$dockerVersion = docker --version 2>&1
if ($dockerVersion) {
    Write-Host "  ✓ $dockerVersion" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Docker 未安装 (可选)" -ForegroundColor Yellow
}

# 检查环境变量
Write-Host "`n环境变量配置:" -ForegroundColor Yellow
if (Test-Path "$backendDir\.env") {
    Write-Host "  ✓ backend/.env 已配置" -ForegroundColor Green
} else {
    Write-Host "  ⚠ backend/.env 不存在，请复制 .env.example" -ForegroundColor Yellow
}

if (Test-Path "$frontendDir\.env") {
    Write-Host "  ✓ frontend/.env 已配置" -ForegroundColor Green
} else {
    Write-Host "  ⚠ frontend/.env 不存在，请复制 .env.example" -ForegroundColor Yellow
}

# 检查 Git
Write-Host "`nGit 状态:" -ForegroundColor Yellow
$gitStatus = git status --porcelain 2>&1
if ($LASTEXITCODE -eq 0) {
    $branch = git branch --show-current 2>&1
    Write-Host "  ✓ 当前分支：$branch" -ForegroundColor Green
    
    if ($gitStatus) {
        Write-Host "  ⚠ 有未提交的更改" -ForegroundColor Yellow
        if ($Detailed) {
            $gitStatus | ForEach-Object { Write-Host "    $_" }
        }
    } else {
        Write-Host "  ✓ 工作区干净" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠ 不是 Git 仓库" -ForegroundColor Yellow
}

# 检查关键文件
Write-Host "`n关键文件:" -ForegroundColor Yellow
$files = @(
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "docker-compose.yml",
    ".github/workflows/ci-cd.yml"
)

foreach ($file in $files) {
    if (Test-Path (Join-Path $PSScriptRoot $file)) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file 缺失" -ForegroundColor Red
    }
}

# 检查测试
Write-Host "`n测试状态:" -ForegroundColor Yellow
$backendTests = Get-ChildItem -Path "$backendDir\tests" -Filter "test_*.py" -ErrorAction SilentlyContinue
if ($backendTests) {
    Write-Host "  ✓ 后端测试文件：$($backendTests.Count) 个" -ForegroundColor Green
} else {
    Write-Host "  ⚠ 未找到后端测试文件" -ForegroundColor Yellow
}

$frontendTests = Get-ChildItem -Path "$frontendDir\src" -Recurse -Filter "*.test.ts*" -ErrorAction SilentlyContinue
if ($frontendTests) {
    Write-Host "  ✓ 前端测试文件：$($frontendTests.Count) 个" -ForegroundColor Green
} else {
    Write-Host "  ⚠ 未找到前端测试文件" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "检查完成!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
