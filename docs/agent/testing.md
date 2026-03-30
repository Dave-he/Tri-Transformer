# 测试规范

## 前端测试（Vitest + Testing Library + MSW）

**框架配置**（`vite.config.ts`）：
```typescript
test: {
  globals: true,
  environment: 'jsdom',
  setupFiles: ['./src/test/testSetup.ts'],
  css: false,
}
```

**测试文件位置**：
- API 客户端测试：`frontend/src/api/__tests__/*.test.ts`
- 组件测试：`frontend/src/components/**/__tests__/*.test.tsx`
- Store 测试：`frontend/src/store/__tests__/*.test.ts`

**运行命令**：
```bash
cd frontend && pnpm test          # 单次运行
cd frontend && pnpm test:watch    # 监听模式
```

**API Mock 约定**：
- MSW handlers 放在 `frontend/src/mocks/handlers/`（按功能拆分）
- 测试中通过 `server.use(...)` 覆盖特定场景
- `mocks/server.ts` 用于 Node 环境（测试），`mocks/browser.ts` 用于浏览器

## 后端测试（pytest + pytest-asyncio + httpx）

**Pytest 配置**（`backend/pyproject.toml`）：
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**测试文件**（`backend/tests/`）：
- `conftest.py` — 全局 fixtures（测试 DB、JWT token、AsyncClient）
- `test_auth.py`, `test_chat.py`, `test_rag.py`, `test_model.py`
- `test_fact_checker.py`, `test_knowledge.py`, `test_stream.py`
- `test_train.py`, `test_tokenizer.py`, `test_pluggable_llm.py`

**运行命令**：
```bash
cd backend && pytest              # 运行所有测试
cd backend && pytest -v           # 详细输出
cd backend && pytest tests/test_chat.py  # 单个文件
cd backend && pytest --cov=app    # 覆盖率报告
```

**关键 Fixtures 约定**：
```python
# 测试数据库使用内存 SQLite，通过 dependency_overrides 注入
app.dependency_overrides[get_db] = override_get_db

# API 测试使用 ASGITransport + AsyncClient
async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
    ...

# 认证 fixtures 返回 JWT token 字符串
headers = {"Authorization": f"Bearer {token}"}
```

## Eval 测试（pytest）

**测试文件**（`eval/tests/`）：
- `test_hallucination_loss.py`
- `test_rag_loss.py`
- `test_control_alignment_loss.py`
- `test_eval_pipeline.py`
- `test_ground_truth_engine.py`

```bash
cd eval && python -m pytest
```

## 通用约定

- 测试覆盖率目标：语句 80%+
- 前端：禁止在测试中使用 `screen.debug()`（会造成噪音）
- 后端：每个测试函数使用独立事务并回滚，保证隔离性
