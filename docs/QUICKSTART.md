# 快速开始指南

> 5 分钟上手 Tri-Transformer 系统

本指南将帮助您快速启动并体验 Tri-Transformer 的核心功能。

---

## 🎯 本指南涵盖

- ✅ 使用 Docker Compose 一键部署
- ✅ 创建第一个知识库文档
- ✅ 体验 RAG 对话
- ✅ 查看模型推理性能

---

## 🚀 快速启动（Docker 方式 - 推荐）

### 步骤 1：克隆项目

```bash
git clone https://github.com/your-org/tri-transformer.git
cd tri-transformer
```

### 步骤 2：配置环境变量

```bash
# 复制后端配置
cp backend/.env.example backend/.env

# 复制前端配置
cp frontend/.env.example frontend/.env.local
```

### 步骤 3：一键启动

```bash
# 构建并启动所有服务（后台运行）
docker-compose up -d

# 查看启动日志
docker-compose logs -f
```

等待 2-3 分钟，所有服务启动完成后：

- 🌐 **前端界面**: http://localhost:3000
- 📚 **API 文档**: http://localhost:8000/docs
- ❤️ **健康检查**: http://localhost:8000/health

### 步骤 4：验证服务

```bash
# 检查后端健康状态
curl http://localhost:8000/health

# 应返回：{"status": "healthy"}
```

---

## 🎨 使用指南

### 1. 注册账号

访问 http://localhost:3000，点击"注册"：

- 用户名：`admin`
- 邮箱：`admin@example.com`
- 密码：`Admin123!@#`

### 2. 上传知识库文档

#### 方法 A：通过前端界面

1. 登录后，导航到 **"知识库"** 页面
2. 点击 **"上传文档"** 按钮
3. 选择文件（支持 PDF、TXT、MD、DOCX）
4. 等待处理完成（状态变为"已索引"）

#### 方法 B：通过 API

```bash
# 获取 JWT Token
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"Admin123!@#"}' \
  | jq -r '.access_token')

# 上传文档
curl -X POST "http://localhost:8000/api/v1/knowledge/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./sample_document.pdf" \
  -F "title=示例文档" \
  -F "description=这是一个测试文档"
```

### 3. 开始 RAG 对话

#### 前端界面

1. 导航到 **"对话"** 页面
2. 在输入框中输入问题，例如：
   - "文档中提到了什么主要内容？"
   - "总结文档的关键点"
3. 查看 AI 回复和引用来源

#### API 调用

```bash
# 创建对话会话
SESSION_ID=$(curl -X POST "http://localhost:8000/api/v1/chat/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  | jq -r '.id')

# 发送消息
curl -X POST "http://localhost:8000/api/v1/chat/sessions/$SESSION_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "文档中提到了什么主要内容？",
    "mode": "rag"
  }'
```

### 4. 体验流式输出（WebSocket）

```bash
# 使用 wscat 测试 WebSocket（需先安装：npm install -g wscat）
wscat -c "ws://localhost:8000/api/v1/model/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -x '{"message": "请总结文档内容", "stream": true}'
```

---

## 🧪 测试模型性能

### 使用测试脚本

```bash
cd backend

# 运行模型推理测试
python scripts/benchmark_inference.py

# 输出示例：
# Model: Qwen2.5-7B
# Device: cuda
# Prompt tokens: 512
# Generated tokens: 256
# Time to first token: 0.15s
# Total generation time: 2.34s
# Tokens per second: 109.4
```

### 查看性能指标

导航到前端 **"性能指标"** 页面，查看：

- 📊 推理延迟
- 📈 吞吐量（tokens/s）
- 💾 显存使用率
- 🔥 GPU 利用率

---

## 📊 默认配置说明

### 开发环境配置

| 组件 | 配置 |
|------|------|
| **模型** | Qwen2.5-7B（模拟模式） |
| **向量数据库** | ChromaDB（内存模式） |
| **主数据库** | SQLite |
| **推理设备** | CPU（自动检测 GPU） |
| **批次大小** | 16 |

### 切换到 GPU 加速

编辑 `backend/.env`：

```ini
INFERENCE_DEVICE=cuda
INFERENCE_DTYPE=bfloat16
USE_FLASH_ATTENTION=true
```

重启服务：

```bash
docker-compose restart backend
```

---

## 🛠️ 命令行快速参考

### Docker 管理

```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 查看日志
docker-compose logs -f backend

# 重启后端
docker-compose restart backend

# 重新构建
docker-compose up -d --build
```

### 后端命令

```bash
cd backend

# 激活虚拟环境
source venv/bin/activate

# 运行测试
pytest tests/test_chat.py -v

# 运行代码格式化
black app/ tests/

# 检查代码质量
flake8 app/
```

### 前端命令

```bash
cd frontend

# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev

# 运行测试
pnpm test

# 构建生产版本
pnpm build
```

---

## 📝 示例场景

### 场景 1：文档问答

1. 上传一份产品手册 PDF
2. 提问："产品的主要功能有哪些？"
3. 系统返回答案并标注引用来源

### 场景 2：多轮对话

1. 上传技术文档
2. 第一轮："总结文档内容"
3. 第二轮："详细解释第二部分"
4. 第三轮："这个技术和 XXX 有什么区别？"

### 场景 3：事实核查

1. 上传多篇相关文档
2. 提问："文档中是否提到了 XXX 技术？"
3. 系统标记不确定的陈述并提示核查

---

## 🔍 故障排查

### 前端无法访问

```bash
# 检查容器状态
docker-compose ps

# 查看前端日志
docker-compose logs frontend

# 重启前端
docker-compose restart frontend
```

### 后端 API 错误

```bash
# 查看后端日志
docker-compose logs backend

# 进入后端容器
docker-compose exec backend bash

# 手动测试 API
curl http://localhost:8000/health
```

### 数据库连接失败

```bash
# 检查数据库容器
docker-compose ps db

# 重启数据库
docker-compose restart db

# 等待 10 秒后重启后端
sleep 10 && docker-compose restart backend
```

---

## 📚 下一步

完成快速开始后，您可以：

- 📖 阅读 [完整文档](../Tri-Transformer 可控对话与 RAG 知识库增强系统.md)
- 🔧 查看 [安装部署指南](./installation/INSTALLATION.md)
- 🏗️ 了解 [系统架构](../agent/architecture.md)
- 🧪 运行 [测试套件](../agent/testing.md)
- 🤝 参与 [贡献开发](../agent/conventions.md)

---

## 💡 提示

- **首次启动较慢**：模型和向量数据库需要初始化，请耐心等待
- **测试数据**：系统预置了示例文档，可直接体验 RAG 对话
- **性能优化**：生产环境请参考 [性能调优指南](./performance.md)
- **获取帮助**：遇到问题请查看 [FAQ](./faq.md) 或提交 Issue

---

**开始时间**: < 5 分钟  
**难度**: ⭐ 简单  
**前置要求**: Docker 和 Docker Compose

---

**最后更新**: 2026-04-03  
**维护者**: Tri-Transformer Team
