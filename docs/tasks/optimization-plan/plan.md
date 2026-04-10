# Tri-Transformer 项目优化计划

## 概览

基于项目全面分析，共识别出 **14 项优化点**，覆盖安全、性能、工程质量、ML 训练四个维度。

- **P0（必须修复）**：3 项 — 安全漏洞或生产稳定性风险
- **P1（高优先级）**：7 项 — 性能与工程质量
- **P2（优化提升）**：4 项 — 锦上添花

预计总工时：**4 周**

---

## P0 — 安全 / 稳定性（Week 1）

### OPT-01｜修复 CORS 安全配置

| 属性 | 值 |
|------|-----|
| 优先级 | P0 |
| 文件 | `backend/app/main.py` |
| 影响 | 安全 |

**问题**：`allow_origins=["*"]` 与 `allow_credentials=True` 同时存在，违反 CORS 规范，允许任意域名携带凭证访问接口。

**修复方案**：
```python
# backend/app/core/config.py
cors_origins: list[str] = Field(default=["http://localhost:3000"])

# backend/app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)
```

**验收标准**：非白名单域名的跨域请求返回 403。

---

### OPT-02｜添加结构化请求日志

| 属性 | 值 |
|------|-----|
| 优先级 | P0 |
| 文件 | `backend/app/main.py`、新增 `backend/app/core/logging.py` |
| 影响 | 可运维性 |

**问题**：所有路由零日志输出，生产环境出错无法定位，异常被 `except Exception: results = []` 静默吞掉（`knowledge.py:173`）。

**修复方案**：
```python
# backend/app/core/logging.py
import logging, sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    logger = logging.getLogger("tri-transformer")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
```

同步修复所有裸 `except Exception: pass/[]` 为带日志的具体处理。

**验收标准**：每条请求输出包含 method、path、status_code、duration_ms 的 JSON 日志行。

---

### OPT-03｜JWT Secret 强制校验 + 接口限流

| 属性 | 值 |
|------|-----|
| 优先级 | P0 |
| 文件 | `backend/app/core/config.py`、`backend/app/api/v1/auth.py` |
| 影响 | 安全 |

**问题**：
1. 默认 `secret_key = "dev-secret-key-change-in-production"` 在生产环境未被阻断
2. `/auth/login` 无速率限制，存在暴力破解风险

**修复方案**：
```python
# config.py
@validator("secret_key")
def check_secret_key(cls, v):
    if v == "dev-secret-key-change-in-production":
        raise ValueError("生产环境必须设置 SECRET_KEY 环境变量")
    return v

# auth.py — 添加 slowapi 限流
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@router.post("/login")
@limiter.limit("5/minute")
async def login(...): ...
```

**验收标准**：
- 未设置 SECRET_KEY 时服务启动失败并打印明确错误
- 同 IP 登录超过 5 次/分钟返回 429

---

## P1 — 性能 / 工程质量（Week 2–3）

### OPT-04｜数据库关键字段添加索引

| 属性 | 值 |
|------|-----|
| 优先级 | P1 |
| 文件 | `backend/app/models/*.py`、新增 Alembic migration |
| 影响 | 查询性能 |

**问题**：`user_id`、`session_id`、`kb_id` 等高频过滤字段无索引，数据量增大后全表扫描。

**修复方案**：
```python
# backend/app/models/chat_session.py
class ChatSession(Base):
    __table_args__ = (
        Index("idx_chat_session_user_id", "user_id"),
        Index("idx_chat_session_created", "user_id", "created_at"),
    )

# backend/app/models/knowledge_document.py
class KnowledgeDocument(Base):
    __table_args__ = (
        Index("idx_doc_kb_id", "kb_id"),
        Index("idx_doc_kb_created", "kb_id", "created_at"),
    )
```

生成并执行对应 Alembic migration。

**验收标准**：`EXPLAIN ANALYZE` 相关查询显示 Index Scan 而非 Seq Scan。

---

### OPT-05｜前端添加 Error Boundary

| 属性 | 值 |
|------|-----|
| 优先级 | P1 |
| 文件 | 新增 `frontend/src/components/common/ErrorBoundary.tsx`、`frontend/src/App.tsx` |
| 影响 | 用户体验 / 稳定性 |

**问题**：子组件运行时异常会导致整个页面白屏，无任何降级处理。

**修复方案**：
```tsx
// frontend/src/components/common/ErrorBoundary.tsx
class ErrorBoundary extends React.Component<Props, State> {
  state = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info)
  }

  render() {
    if (this.state.hasError) {
      return <Result status="error" title="页面出错了" extra={<Button onClick={() => this.setState({ hasError: false })}>重试</Button>} />
    }
    return this.props.children
  }
}

// App.tsx — 包裹各 Route
<ErrorBoundary>
  <Suspense fallback={<Spin />}>
    <Routes>...</Routes>
  </Suspense>
</ErrorBoundary>
```

**验收标准**：子组件抛出异常时显示错误提示而非白屏，支持"重试"恢复。

---

### OPT-06｜文件上传真实 MIME 类型校验

| 属性 | 值 |
|------|-----|
| 优先级 | P1 |
| 文件 | `backend/app/api/v1/knowledge.py`、`backend/requirements.txt` |
| 影响 | 安全 |

**问题**：仅检查文件扩展名，可通过重命名绕过，上传恶意文件。

**修复方案**：
```python
import magic  # python-magic

ALLOWED_MIMES = {"application/pdf", "text/plain", "text/markdown",
                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

async def upload_document(file: UploadFile, ...):
    content = await file.read()
    detected = magic.Magic(mime=True).from_buffer(content)
    if detected not in ALLOWED_MIMES:
        raise HTTPException(status_code=422, detail=f"不支持的文件类型: {detected}")
```

**验收标准**：上传伪装扩展名的文件返回 422。

---

### OPT-07｜Docker 生产化改造

| 属性 | 值 |
|------|-----|
| 优先级 | P1 |
| 文件 | `backend/Dockerfile`、`frontend/Dockerfile`、`docker-compose.yml` |
| 影响 | 安全 / 可靠性 |

**问题**：
1. 容器以 root 运行
2. 无 HEALTHCHECK
3. Frontend Dockerfile 用 `npm` 而项目使用 `pnpm`
4. docker-compose 无 depends_on 健康检查

**修复方案**：
```dockerfile
# backend/Dockerfile
FROM python:3.10-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --chown=appuser:appuser . .
USER appuser
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1

# frontend/Dockerfile
RUN corepack enable && pnpm install --frozen-lockfile
```

```yaml
# docker-compose.yml
backend:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 10s
    retries: 3
frontend:
  depends_on:
    backend:
      condition: service_healthy
```

**验收标准**：`docker-compose up` 后 `docker ps` 显示所有容器状态为 `healthy`。

---

### OPT-08｜CI/CD 添加安全扫描

| 属性 | 值 |
|------|-----|
| 优先级 | P1 |
| 文件 | `.github/workflows/ci-cd.yml` |
| 影响 | 安全 |

**问题**：每次发布无依赖漏洞扫描，存在已知 CVE 未被发现的风险。

**修复方案**：
```yaml
security-scan:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Trivy 文件系统扫描
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: fs
        exit-code: 1
        severity: CRITICAL,HIGH
    - name: Bandit Python SAST
      run: pip install bandit && bandit -r backend/app -ll -f json -o bandit.json
    - name: pnpm audit
      run: cd frontend && pnpm audit --audit-level high
```

**验收标准**：存在 HIGH/CRITICAL 漏洞时 CI 流水线失败阻断合并。

---

### OPT-09｜后端安全响应头

| 属性 | 值 |
|------|-----|
| 优先级 | P1 |
| 文件 | `backend/app/main.py` |
| 影响 | 安全 |

**问题**：缺少 `X-Frame-Options`、`X-Content-Type-Options` 等安全头，存在点击劫持风险。

**修复方案**：
```python
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

**验收标准**：curl 响应头包含上述四个安全头。

---

### OPT-10｜RAG 嵌入结果缓存

| 属性 | 值 |
|------|-----|
| 优先级 | P1 |
| 文件 | `backend/app/services/rag/embedder.py` |
| 影响 | 响应性能 |

**问题**：相同查询反复调用 embedding 模型，无任何缓存，高并发下延迟高。

**修复方案**：
```python
from functools import lru_cache
import hashlib

class BGEEmbedder(BaseEmbedder):
    def __init__(self):
        self._cache: dict[str, list[float]] = {}

    async def embed(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = await self._do_embed(text)
        return self._cache[key]
```

后续可升级为 Redis 分布式缓存。

**验收标准**：相同查询第二次响应时间 < 10ms（vs 首次 ~200ms）。

---

## P2 — 优化提升（Week 4）

### OPT-11｜ML 训练效率：AMP + 梯度累积

| 属性 | 值 |
|------|-----|
| 优先级 | P2 |
| 文件 | `backend/app/model/trainer.py` |
| 影响 | 训练速度 / 显存占用 |

**问题**：无混合精度训练（AMP），无梯度累积，小 GPU 上训练效率低。

**修复方案**：
```python
from torch.cuda.amp import autocast, GradScaler

class TriTransformerTrainer:
    def __init__(self, gradient_accumulation_steps: int = 4):
        self.scaler = GradScaler()
        self.grad_accum_steps = gradient_accumulation_steps

    def train_step(self, batch, step):
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss / self.grad_accum_steps

        self.scaler.scale(loss).backward()

        if (step + 1) % self.grad_accum_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
```

**验收标准**：同等配置下显存占用减少 ~40%，训练吞吐提升 ~30%。

---

### OPT-12｜前端 Bundle 分析配置

| 属性 | 值 |
|------|-----|
| 优先级 | P2 |
| 文件 | `frontend/vite.config.ts`、`frontend/package.json` |
| 影响 | 构建可见性 |

**修复方案**：
```typescript
// vite.config.ts
import { visualizer } from "rollup-plugin-visualizer"

export default defineConfig({
  plugins: [
    react(),
    visualizer({ open: true, filename: "dist/bundle-stats.html" }),
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
          antd: ["antd"],
          charts: ["recharts"],
        },
      },
    },
  },
})
```

**验收标准**：`pnpm build` 后生成 `dist/bundle-stats.html` 可视化报告。

---

### OPT-13｜评测对齐标准基准数据集

| 属性 | 值 |
|------|-----|
| 优先级 | P2 |
| 文件 | `eval/pipeline/`、新增 `eval/benchmarks/` |
| 影响 | 模型质量可信度 |

**问题**：仅有自研评测流程，无法与业界对比；模型改进无量化基准。

**修复方案**：
- 集成 SQuAD 2.0（阅读理解基准）
- 集成 HotpotQA（多跳推理基准）
- 添加 TruthfulQA（幻觉检测基准）

```python
# eval/benchmarks/squad_eval.py
class SQuADEvaluator:
    def evaluate(self, model) -> dict:
        # 返回 EM、F1 分数
```

**验收标准**：`eval/benchmarks/` 下三个数据集均可一键运行并输出对比报告。

---

### OPT-14｜学习率调度

| 属性 | 值 |
|------|-----|
| 优先级 | P2 |
| 文件 | `backend/app/model/trainer.py` |
| 影响 | 训练收敛质量 |

**修复方案**：
```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=config.learning_rate,
    total_steps=total_steps,
    pct_start=0.1,       # 10% warm-up
    anneal_strategy="cos",
)
```

**验收标准**：训练 loss 曲线更平稳，最终收敛值优于固定 lr 基线。

---

## 执行时间表

```
Week 1  ████████████████  OPT-01 CORS | OPT-02 日志 | OPT-03 JWT+限流
Week 2  ████████████████  OPT-04 DB索引 | OPT-05 ErrorBoundary | OPT-06 文件校验
Week 3  ████████████████  OPT-07 Docker | OPT-08 CI安全扫描 | OPT-09 安全头 | OPT-10 缓存
Week 4  ████████████████  OPT-11 AMP训练 | OPT-12 Bundle | OPT-13 基准 | OPT-14 LR调度
```

## 验收指标

| 指标 | 当前 | 目标 |
|------|------|------|
| 高危安全漏洞 | 未知 | 0 |
| API P95 响应时间 | 未监控 | < 200ms |
| 相同查询 embedding 延迟 | ~200ms | < 10ms（命中缓存） |
| 容器健康检查 | 无 | 全部 healthy |
| 训练显存占用 | baseline | 减少 ~40% |
| CI 安全扫描 | 无 | 阻断 HIGH/CRITICAL |
