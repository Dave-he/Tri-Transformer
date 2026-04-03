# 安装部署指南

本指南详细说明 Tri-Transformer 系统的安装、配置和部署流程。

---

## 📋 目录

- [系统要求](#-系统要求)
- [环境准备](#-环境准备)
- [后端安装](#-后端安装)
- [前端安装](#-前端安装)
- [Docker 部署](#-docker-部署)
- [生产环境配置](#-生产环境配置)
- [故障排查](#-故障排查)

---

## 🖥️ 系统要求

### 最低配置

| 组件 | 要求 |
|------|------|
| **操作系统** | Linux (Ubuntu 20.04+) / macOS 12+ / Windows 11 (WSL2) |
| **CPU** | 8 核心（推荐 16 核心+） |
| **内存** | 32GB RAM（推荐 64GB+） |
| **GPU** | NVIDIA GPU 16GB+ 显存（GTX 4090 / RTX A6000 或更高） |
| **存储** | 100GB 可用空间（SSD 推荐） |

### 推荐配置（生产环境）

| 组件 | 要求 |
|------|------|
| **操作系统** | Ubuntu 22.04 LTS |
| **CPU** | 32 核心 AMD EPYC / Intel Xeon |
| **内存** | 128GB RAM |
| **GPU** | NVIDIA A100 40GB/80GB × 2-8 |
| **存储** | 1TB NVMe SSD |
| **网络** | 10GbE 网络接口 |

### 软件依赖

| 软件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| **Python** | 3.10 | 3.10+ |
| **Node.js** | 18 | 20+ |
| **pnpm** | 8 | 9+ |
| **Docker** | 20.10 | 24+ |
| **Docker Compose** | 2.0 | 2.20+ |
| **CUDA** | 11.8 | 12.1+ |
| **NVIDIA Driver** | 520+ | 535+ |

---

## 🔧 环境准备

### 1. 验证系统依赖

```bash
# 检查 Python 版本
python --version  # 应 >= 3.10

# 检查 pip 版本
pip --version  # 应 >= 22.0

# 检查 Node.js 版本
node --version  # 应 >= 18

# 检查 pnpm 版本
pnpm --version  # 应 >= 8

# 检查 Docker（可选）
docker --version
docker-compose --version

# 检查 CUDA（GPU 加速必需）
nvidia-smi
nvcc --version
```

### 2. 安装系统级依赖（Ubuntu/Debian）

```bash
# 更新包索引
sudo apt update

# 安装基础工具
sudo apt install -y \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# 安装 GPU 驱动（如未安装）
# 参考：https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes.html
```

### 3. 安装 Python 3.10+（如系统版本过低）

```bash
# 使用 deadsnakes PPA（Ubuntu）
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# 设置默认 Python 版本
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
```

### 4. 安装 Node.js 20+

```bash
# 使用 NodeSource 仓库
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# 或使用 nvm（推荐）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20
```

### 5. 安装 pnpm

```bash
# 使用 npm 安装
npm install -g pnpm

# 验证安装
pnpm --version
```

---

## 🐍 后端安装

### 1. 克隆项目

```bash
git clone https://github.com/your-org/tri-transformer.git
cd tri-transformer
```

### 2. 创建 Python 虚拟环境

```bash
cd backend

# 方法 1：使用 venv（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate  # Windows

# 方法 2：使用 conda（可选）
conda create -n tri-transformer python=3.10
conda activate tri-transformer
```

### 3. 安装 Python 依赖

```bash
# 升级 pip
pip install --upgrade pip setuptools wheel

# 安装核心依赖
pip install -r requirements.txt

# 开发环境额外依赖
pip install -r requirements-dev.txt  # 如存在
```

### 4. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置文件
nano .env  # 或使用你喜欢的编辑器
```

#### .env 配置说明

```ini
# ========== 应用配置 ==========
APP_NAME=Tri-Transformer
ENV=development  # development / production
DEBUG=True
SECRET_KEY=your-secret-key-here-change-in-production

# ========== 数据库配置 ==========
# 开发环境使用 SQLite
DATABASE_URL=sqlite+aiosqlite:///./tri_transformer.db

# 生产环境使用 PostgreSQL
# DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/tri_transformer

# ========== JWT 配置 ==========
JWT_SECRET_KEY=your-jwt-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# ========== 模型配置 ==========
# 骨干模型路径（本地或 HuggingFace）
MODEL_A_NAME=Qwen/Qwen2.5-7B
MODEL_B_NAME=Qwen/Qwen2.5-7B
MODEL_CACHE_DIR=./models_cache

# 推理设备
INFERENCE_DEVICE=cuda  # cuda / cpu
INFERENCE_DTYPE=bfloat16  # float16 / bfloat16 / float32

# ========== RAG 配置 ==========
# 向量数据库
VECTOR_STORE=chromadb  # chromadb / milvus
CHROMA_PERSIST_DIR=./chroma_db

# Milvus 配置（如使用 Milvus）
# MILVUS_URI=http://localhost:19530
# MILVUS_TOKEN=your_milvus_token

# 嵌入模型
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# ========== 日志配置 ==========
LOG_LEVEL=INFO  # DEBUG / INFO / WARNING / ERROR
LOG_FILE=./logs/app.log

# ========== CORS 配置（前端地址） ==========
FRONTEND_URL=http://localhost:3000
```

### 5. 初始化数据库

```bash
# 运行数据库迁移（如使用 Alembic）
# alembic upgrade head

# 或手动初始化
python -c "from app.core.database import create_tables; create_tables()"
```

### 6. 启动后端服务

```bash
# 开发模式（热重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式（多进程）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用 gunicorn（推荐生产环境）
gunicorn app.main:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 5
```

访问 http://localhost:8000/docs 查看 API 文档。

### 7. 验证后端安装

```bash
# 运行测试
pytest

# 运行特定测试
pytest tests/test_model.py -v

# 检查代码质量
flake8 app/ tests/
black --check app/ tests/
```

---

## ⚛️ 前端安装

### 1. 安装依赖

```bash
cd frontend

# 使用 pnpm 安装（推荐）
pnpm install

# 或使用 npm
# npm install

# 或使用 yarn
# yarn install
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env.local

# 编辑配置
nano .env.local
```

#### .env.local 配置说明

```ini
# 后端 API 地址
VITE_API_BASE_URL=http://localhost:8000/api/v1

# WebSocket 地址
VITE_WS_URL=ws://localhost:8000/api/v1/model/stream

# WebRTC 信令服务器地址
VITE_WEBRTC_SIGNALING_URL=ws://localhost:8000/api/v1/webrtc

# 应用配置
VITE_APP_NAME=Tri-Transformer
VITE_APP_VERSION=1.0.0

# 功能开关
VITE_ENABLE_MOCK=false
VITE_ENABLE_DEBUG=true
```

### 3. 启动开发服务器

```bash
# 开发模式
pnpm dev

# 指定端口
pnpm dev --port 3000

# 暴露到局域网
pnpm dev --host
```

访问 http://localhost:3000 查看前端界面。

### 4. 构建生产版本

```bash
# 生产构建
pnpm build

# 预览生产构建
pnpm preview

# 分析打包体积
pnpm build -- --mode analyze
```

### 5. 代码质量检查

```bash
# ESLint 检查
pnpm lint

# TypeScript 类型检查
pnpm typecheck

# 运行测试
pnpm test

# 测试覆盖率
pnpm test:coverage
```

---

## 🐳 Docker 部署

### 1. 安装 Docker 和 Docker Compose

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl enable docker
sudo systemctl start docker

# 安装 Docker Compose Plugin
sudo apt install docker-compose-plugin

# 验证安装
docker --version
docker compose version
```

### 2. 配置 Docker 环境

```bash
# 复制 Docker 配置示例
cp docker-compose.yml.example docker-compose.yml

# 编辑配置（如需要）
nano docker-compose.yml
```

### 3. Docker Compose 配置说明

```yaml
version: '3.8'

services:
  # ========== 后端服务 ==========
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: tri-transformer-backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - DATABASE_URL=postgresql+asyncpg://tri_transformer:password@db:5432/tri_transformer
      - VECTOR_STORE=milvus
      - MILVUS_URI=http://milvus:19530
    volumes:
      - ./backend/models_cache:/app/models_cache
      - ./backend/logs:/app/logs
    depends_on:
      - db
      - milvus
    networks:
      - tri-transformer-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  # ========== PostgreSQL 数据库 ==========
  db:
    image: postgres:15-alpine
    container_name: tri-transformer-db
    restart: always
    environment:
      POSTGRES_DB: tri_transformer
      POSTGRES_USER: tri_transformer
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - tri-transformer-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tri_transformer"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ========== Milvus 向量数据库 ==========
  milvus:
    image: milvusdb/milvus:v2.4.0
    container_name: tri-transformer-milvus
    restart: unless-stopped
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    depends_on:
      - etcd
      - minio
    networks:
      - tri-transformer-net
    ports:
      - "19530:19530"
      - "9091:9091"

  # ========== etcd（Milvus 依赖） ==========
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: tri-transformer-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    networks:
      - tri-transformer-net

  # ========== MinIO（Milvus 对象存储） ==========
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: tri-transformer-minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    networks:
      - tri-transformer-net
    ports:
      - "9000:9000"
      - "9001:9001"

  # ========== 前端服务 ==========
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        - VITE_API_BASE_URL=http://localhost:8000/api/v1
    container_name: tri-transformer-frontend
    restart: unless-stopped
    ports:
      - "3000:80"
    depends_on:
      - backend
    networks:
      - tri-transformer-net

  # ========== Nginx 反向代理（可选） ==========
  nginx:
    image: nginx:alpine
    container_name: tri-transformer-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - frontend
      - backend
    networks:
      - tri-transformer-net

volumes:
  postgres_data:
  milvus_data:
  etcd_data:
  minio_data:

networks:
  tri-transformer-net:
    driver: bridge
```

### 4. 启动所有服务

```bash
# 构建并启动
docker compose up -d

# 查看构建日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f backend
docker compose logs -f frontend
```

### 5. 管理 Docker 服务

```bash
# 停止所有服务
docker compose down

# 停止并删除数据卷（危险操作！）
docker compose down -v

# 重启服务
docker compose restart

# 重新构建并启动
docker compose up -d --build

# 查看服务状态
docker compose ps

# 进入容器
docker compose exec backend bash
docker compose exec frontend sh
```

### 6. GPU 支持（Docker）

```bash
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证 GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 生产环境配置

### 1. 安全配置

```bash
# 生成安全的 SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 配置 HTTPS（使用 Let's Encrypt）
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 2. 性能优化

#### 后端优化

```ini
# .env 生产配置
ENV=production
DEBUG=False
LOG_LEVEL=WARNING

# 数据库连接池
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# 模型优化
INFERENCE_BATCH_SIZE=32
USE_FLASH_ATTENTION=true
USE_VLLM=true
```

#### 前端优化

```ini
# .env.local 生产配置
VITE_ENABLE_DEBUG=false
VITE_ENABLE_MOCK=false

# 启用 CDN
VITE_CDN_URL=https://cdn.your-domain.com
```

### 3. 监控与日志

```bash
# 安装 Prometheus + Grafana（可选）
docker compose -f docker-compose.monitoring.yml up -d

# 配置日志轮转
# /etc/logrotate.d/tri-transformer
/var/log/tri-transformer/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
}
```

### 4. 备份策略

```bash
#!/bin/bash
# backup.sh

# 备份数据库
pg_dump -h localhost -U tri_transformer tri_transformer > backup_db_$(date +%Y%m%d).sql

# 备份向量数据库
tar -czf backup_milvus_$(date +%Y%m%d).tar.gz /var/lib/milvus

# 备份到远程存储
rsync -avz backup_* user@backup-server:/backups/tri-transformer/

# 清理 7 天前的备份
find /backups/tri-transformer -name "backup_*" -mtime +7 -delete
```

---

## 🔍 故障排查

### 常见问题

#### 1. 后端无法启动

```bash
# 检查端口占用
lsof -i :8000

# 查看日志
tail -f backend/logs/app.log

# 验证依赖
pip check

# 测试数据库连接
python -c "from app.core.database import engine; print(engine.url)"
```

#### 2. 前端构建失败

```bash
# 清理缓存
pnpm clean
rm -rf node_modules
pnpm install

# 检查 Node.js 版本
node --version  # 应 >= 18

# 查看详细错误
pnpm build --debug
```

#### 3. GPU 不可用

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 验证 PyTorch GPU 支持
python -c "import torch; print(torch.cuda.is_available())"

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

#### 4. 向量数据库连接失败

```bash
# 检查 ChromaDB
ls -la chroma_db/

# 检查 Milvus
docker compose ps milvus
docker compose logs milvus

# 测试连接
python -c "from pymilvus import connections; connections.connect()"
```

#### 5. 内存不足

```bash
# 监控 GPU 显存
watch -n 1 nvidia-smi

# 监控系统内存
htop

# 减小批次大小
# 编辑 .env：INFERENCE_BATCH_SIZE=16
```

### 获取帮助

- 📖 查看 [FAQ 文档](./faq.md)
- 💬 提交 [GitHub Issue](https://github.com/your-org/tri-transformer/issues)
- 📧 联系技术支持：support@example.com

---

## 📞 下一步

安装完成后，请参阅：

- [快速开始指南](../quickstart.md) - 使用 Tri-Transformer 进行对话
- [API 文档](../api-reference.md) - 完整的 API 接口说明
- [架构文档](../agent/architecture.md) - 深入了解系统架构

---

**最后更新**: 2026-04-03
**维护者**: Tri-Transformer Team
