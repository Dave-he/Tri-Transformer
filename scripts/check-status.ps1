# Tri-Transformer Project Status Check Script

param([switch]$Detailed)

$ErrorActionPreference = "SilentlyContinue"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "`n========================================"
Write-Host "Tri-Transformer Status Check"
Write-Host "========================================`n"

# Python
Write-Host "Python Environment:"
try {
    $py = python --version 2>&1
    if ($py) { Write-Host "  OK $py" -ForegroundColor Green } else { Write-Host "  X Python not found" -ForegroundColor Red }
} catch { Write-Host "  X Python not found" -ForegroundColor Red }

# Node.js
Write-Host "`nNode.js Environment:"
try {
    $node = node --version 2>&1
    if ($node) { Write-Host "  OK Node.js $node" -ForegroundColor Green } else { Write-Host "  X Node.js not found" -ForegroundColor Red }
} catch { Write-Host "  X Node.js not found" -ForegroundColor Red }

# Backend
Write-Host "`nBackend Status:"
$be = "$PROJECT_ROOT\..\backend"
if (Test-Path "$be\__pycache__") { 
    Write-Host "  OK Dependencies installed" -ForegroundColor Green 
} else { 
    Write-Host "  ! Need to install dependencies" -ForegroundColor Yellow 
}
if (Test-Path "$be\.env") { 
    Write-Host "  OK .env configured" -ForegroundColor Green 
} else { 
    Write-Host "  ! .env not found, copy from .env.example" -ForegroundColor Yellow 
}

# Frontend
Write-Host "`nFrontend Status:"
$fe = "$PROJECT_ROOT\..\frontend"
if (Test-Path "$fe\node_modules") { 
    Write-Host "  OK node_modules installed" -ForegroundColor Green 
} else { 
    Write-Host "  ! Need to run npm install" -ForegroundColor Yellow 
}
if (Test-Path "$fe\.env") { 
    Write-Host "  OK .env configured" -ForegroundColor Green 
} else { 
    Write-Host "  ! .env not found, copy from .env.example" -ForegroundColor Yellow 
}

# Git
Write-Host "`nGit Status:"
try {
    $branch = git branch --show-current 2>&1
    Write-Host "  OK Branch: $branch" -ForegroundColor Green
    $status = git status --porcelain 2>&1
    if ($status) { 
        Write-Host "  ! Uncommitted changes exist" -ForegroundColor Yellow 
    } else { 
        Write-Host "  OK Working tree clean" -ForegroundColor Green 
    }
} catch { Write-Host "  ! Not a git repository" -ForegroundColor Yellow }

# Key Files
Write-Host "`nKey Files:"
@("README.md","LICENSE","CONTRIBUTING.md","docker-compose.yml") | ForEach-Object {
    if (Test-Path "$PROJECT_ROOT\..\$_") { 
        Write-Host "  OK $_" -ForegroundColor Green 
    } else { 
        Write-Host "  X $_ missing" -ForegroundColor Red 
    }
}

# Tests
Write-Host "`nTest Files:"
$beTests = (Get-ChildItem -Path "$be\tests" -Filter "test_*.py" -ErrorAction SilentlyContinue).Count
Write-Host "  OK Backend: $beTests test files" -ForegroundColor Green
$feTests = (Get-ChildItem -Path "$fe\src" -Recurse -Filter "*.test.ts*" -ErrorAction SilentlyContinue).Count
Write-Host "  OK Frontend: $feTests test files" -ForegroundColor Green

Write-Host "`n========================================"
Write-Host "Status check completed!"
Write-Host "========================================`n"
