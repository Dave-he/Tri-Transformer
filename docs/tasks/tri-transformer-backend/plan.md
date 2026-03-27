# Tri-Transformer 后端服务 - 任务清单

**任务 ID**: tri-transformer-backend
**平台**: Python / FastAPI
**创建时间**: 2026-03-27
**测试命令**: `cd backend && pytest tests/ -v --cov=app --cov-report=term-missing`
**Lint 命令**: `cd backend && black --check app/ tests/ && flake8 app/ tests/`

---

## 任务统计

| 类型 | P0 | P1 | 合计 |
|------|----|----|------|
| test | 6  | 1  | 7    |
| code | 6  | 1  | 7    |
| **合计** | **12** | **2** | **14** |

---

## 执行顺序（测试优先 TDD）

### [RED] 阶段 - 先写所有测试

| ID | 标题 | 优先级 | 依赖 |
|----|------|--------|------|
| T0-1 | 为项目基础结构编写测试 fixtures | P0 | - |
| T1-1 | 为用户认证模块编写单元测试 | P0 | T0-1 |
| T2-1 | 为 RAG 引擎编写单元测试（Mock 嵌入） | P0 | T0-1 |
| T3-1 | 为知识库管理 API 编写测试 | P0 | T0-1, T1-1 |
| T4-1 | 为推理接口编写测试（Mock 模式） | P0 | T0-1, T1-1 |
| T5-1 | 为对话管理 API 编写测试 | P0 | T0-1, T1-1 |
| T6-1 | 为训练调度 API 编写测试 | P1 | T0-1, T1-1 |

### [GREEN] 阶段 - 逐模块实现代码

| ID | 标题 | 优先级 | 依赖 |
|----|------|--------|------|
| P0-1 | 实现项目基础结构（配置/数据库/安全工具） | P0 | T0-1 |
| P0-2 | 实现用户 ORM 模型和认证路由 | P0 | T1-1, P0-1 |
| P0-3 | 实现 RAG 引擎（文档处理/嵌入/向量存储/检索/重排） | P0 | T2-1, P0-1 |
| P0-4 | 实现知识库管理 API（文档 CRUD/多租户） | P0 | T3-1, P0-2, P0-3 |
| P0-5 | 实现 Tri-Transformer 推理接口（含 Mock 实现） | P0 | T4-1, P0-2 |
| P0-6 | 实现对话管理服务和 API（串联完整链路） | P0 | T5-1, P0-3, P0-4, P0-5 |
| P1-1 | 实现训练调度服务和 API | P1 | T6-1, P0-2 |

---

## 任务详情

### T0-1: 测试基础 fixtures

**文件**: `backend/tests/conftest.py`

**Given/When/Then**:
- Given: 测试环境启动
- When: 执行任意测试
- Then:
  - async test client 可用
  - in-memory SQLite DB 可用
  - mock JWT token 可用

---

### T1-1: 认证模块测试

**文件**: `backend/tests/test_auth.py`

**测试用例**:
1. `test_register_success` - 注册成功返回 201
2. `test_register_duplicate` - 重复注册返回 409
3. `test_login_success` - 登录返回 JWT
4. `test_login_wrong_password` - 错误密码返回 401
5. `test_protected_route_without_token` - 无 token 返回 401

---

### T2-1: RAG 引擎测试

**文件**: `backend/tests/test_rag.py`

**测试用例**:
1. `test_document_processor_pdf` - PDF 文本提取
2. `test_document_processor_markdown` - MD 解析
3. `test_chunker_fixed_window` - 固定窗口分块
4. `test_embedder_mock` - Mock 嵌入返回正确维度
5. `test_vector_store_kb_isolation` - kb_id 隔离
6. `test_retriever_returns_topk` - 检索 Top-K

---

### T3-1: 知识库管理测试

**文件**: `backend/tests/test_knowledge.py`

**测试用例**:
1. `test_upload_document` - 上传 PDF 成功
2. `test_list_documents_isolation` - kb_id 隔离列表
3. `test_delete_document` - 删除文档
4. `test_search_knowledge` - 检索接口
5. `test_unauthorized_access` - 未认证返回 401

---

### T4-1: 推理接口测试

**文件**: `backend/tests/test_model.py`

**测试用例**:
1. `test_inference_mock_mode` - Mock 模式推理成功
2. `test_inference_retry_on_failure` - 失败重试逻辑
3. `test_inference_missing_params` - 参数缺失 422
4. `test_inference_unauthorized` - 未认证 401

---

### T5-1: 对话管理测试

**文件**: `backend/tests/test_chat.py`

**测试用例**:
1. `test_create_session` - 创建会话
2. `test_send_message_with_sources` - 发消息返回来源引用
3. `test_get_history` - 获取对话历史
4. `test_cross_user_session_forbidden` - 跨用户 403

---

### T6-1: 训练调度测试

**文件**: `backend/tests/test_train.py`

**测试用例**:
1. `test_submit_train_job` - 提交任务
2. `test_get_job_status` - 查询状态
3. `test_cancel_job` - 取消任务
4. `test_job_not_found` - 不存在任务 404

---

## 验收标准

- [ ] 所有 P0 API 端点返回正确 HTTP 状态码
- [ ] JWT 认证保护所有业务接口，未授权返回 401
- [ ] MOCK_INFERENCE=true 环境下完整对话链路可用
- [ ] 知识库按 kb_id 多租户隔离
- [ ] `pytest tests/ -v` 全部通过，覆盖率 >= 80%
