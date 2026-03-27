# 任务计划: Tri-Transformer 后端服务开发

## 目标
基于 FastAPI/Python 构建 Tri-Transformer 可控对话与 RAG 知识库增强系统后端服务，
涵盖 RAG 引擎、对话管理、知识库 CRUD、推理接口、JWT 认证、训练调度共 9 个功能模块。

## 当前阶段
Phase 2: 技术方案设计

## 阶段规划

### Phase 1: 需求与发现
- [x] 读取 PRD 文档，理解后端范围
- [x] 提取 9 个功能模块的 AC
- [x] 生成 requirement.yaml + requirement.md
- **Status:** complete

### Phase 2: 技术方案设计（Stage 3）
- [ ] 设计目录结构（FastAPI 模块化）
- [ ] 设计核心 API 路由（RESTful）
- [ ] 设计 RAG 流水线（LlamaIndex）
- [ ] 设计数据模型（ORM + 向量库）
- [ ] 生成 tech-solution.yaml + tech-solution.md
- **Status:** in_progress

### Phase 3: 任务拆解（Stage 5）
- [ ] 拆分为 TDD 可执行任务（test → code 依赖链）
- [ ] 生成 plan.yaml + plan.md
- **Status:** pending

### Phase 4: 方案验证（Stage 6）
- [ ] 验证 plan.yaml 覆盖所有 AC
- [ ] 确认 test.command 字段非空
- [ ] 生成 verification-report.md
- **Status:** pending

### Phase 5: TDD 实现（Stage 7）
- [ ] [RED] 先写所有测试文件
- [ ] [GREEN] 逐模块实现代码
- [ ] [REFACTOR] 重构优化
- 实现顺序: 项目结构 → 认证 → 知识库管理 → RAG引擎 → 对话管理 → 推理接口 → 训练调度 → 后处理
- **Status:** pending

### Phase 6: 变更审查与验收（Stage 7.6）
- [ ] 文件变更审查
- [ ] pytest 运行验证
- **Status:** pending

## 关键决策记录

| 决策 | 理由 |
|------|------|
| FastAPI + Python 3.10+ | PRD 明确指定，AI/ML 生态最佳适配 |
| LlamaIndex 作为 RAG 引擎 | PRD 指定，比 LangChain 更专注文档检索 |
| Chroma 向量数据库（MVP） | 轻量，无需独立部署，适合个人/开发环境 |
| PostgreSQL 元数据 | 关系型数据，支持多租户 kb_id 隔离 |
| JWT 认证（python-jose）| 无状态，适合 API 服务 |
| Mock 推理模式 | 无 GPU 环境也能开发测试 |
| pytest + pytest-asyncio | FastAPI 异步测试标准方案 |

## 核心关注点

1. **RAG 流水线设计**: 文档上传 → 分块 → BGE 嵌入 → 存 Chroma → 检索 → 重排的完整流水线
2. **推理接口 Mock**: Tri-Transformer 模型在无 GPU 时需有 Mock 实现
3. **多租户隔离**: 所有知识库操作需按 kb_id 隔离
4. **异步处理**: 文档摄入和训练任务需异步执行（AsyncIO + 后台任务）
5. **错误处理**: 推理失败自动重试 3 次，友好错误信息

## 错误记录

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------| 
| rd-workflow 升级失败 | 1 | 使用现有版本 0.3.50 继续 |
