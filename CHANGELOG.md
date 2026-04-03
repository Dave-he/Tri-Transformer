# 变更日志 (CHANGELOG)

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [未发布]

### 计划功能

#### 新增
- 🎯 **Phase 2: 音频对话支持**
  - EnCodec/SNAC 音频 Token 化
  - 实时语音流处理
  - 低延迟语音对话（目标 < 500ms）

- 🎯 **Phase 3: 全模态数字人**
  - VQ-GAN 视频 Token 化
  - WebRTC 全双工通信
  - 多模态数字人交互

- 🔧 **模型优化**
  - Qwen3-30B-A3B MoE 支持
  - 4bit 量化推理（QLoRA）
  - 多 GPU 张量并行

- 📚 **RAG 增强**
  - 多模态检索（图像/音频）
  - GraphRAG 知识图谱融合
  - 混合检索策略优化

#### 改进
- ⚡ 性能优化
  - FlashAttention-3 集成
  - vLLM 连续批处理
  - KV Cache 优化

- 🎨 用户体验
  - 前端响应速度提升
  - 流式输出优化
  - 错误提示改进

---

## [1.0.0] - 2026-04-03

### ✨ 新增功能

#### 核心架构
- 🎉 **Tri-Transformer 三分支模型**
  - I-Transformer（正向 Decoder-Encoder）实时输入编码
  - C-Transformer（DiT 控制中枢）全局生成控制
  - O-Transformer（反向 Encoder-Decoder）实时输出解码
  - 深度扭合机制实现可控生成

- 🔌 **双端大模型插拔系统**
  - 支持 Qwen3-8B/14B/32B Dense 模型
  - 支持 Qwen3-30B-A3B MoE 模型
  - 左右端异构模型"缝合"训练
  - LoRA 适配器轻量微调

- 🧠 **Thinking Mode**
  - 动态切换思考/非思考模式
  - 复杂推理能力增强
  - 内部 Chain-of-Thought 支持

#### RAG 知识库
- 📚 **文档管理系统**
  - 多格式文档上传（PDF/TXT/MD/DOCX）
  - 自动分块与索引
  - 文档状态追踪

- 🔍 **智能检索**
  - 向量检索（ChromaDB/Milvus）
  - BM25 重排序
  - 混合检索策略
  - 多模态检索准备

- 🎯 **幻觉检测**
  - 自定义幻觉损失函数
  - 实时事实核查
  - 无幻觉阻断机制
  - C-RAG 风险量化

#### 前后端应用
- ⚛️ **React 前端**
  - 现代化 UI（Ant Design）
  - Zustand 状态管理
  - 实时对话界面
  - 知识库管理面板
  - 训练监控仪表板

- ⚡ **FastAPI 后端**
  - RESTful API 设计
  - JWT 认证授权
  - 异步数据库操作
  - WebSocket 流式输出
  - OpenAPI 文档

- 🎛️ **训练系统**
  - LoRA 微调支持
  - 训练任务管理
  - 进度实时监控
  - 指标记录与可视化

#### 基础设施
- 🐳 **Docker 部署**
  - Docker Compose 一键启动
  - Milvus 向量数据库
  - PostgreSQL 数据库
  - Nginx 反向代理

- 📊 **评估管线**
  - 自定义损失函数评估
  - RAG 效果评估
  - 幻觉检测评估
  - CI 门禁系统

### 🔧 技术特性

#### 性能优化
- FlashAttention-2/3 支持
- DeepSpeed ZeRO-3 分布式训练
- vLLM 高效推理（规划中）
- PagedAttention KV Cache 管理
- 连续批处理

#### 模型特性
- QK-Norm 训练稳定化
- GQA（Grouped Query Attention）
- RoPE θ=1,000,000 长上下文支持
- adaLN-Zero 可控调制
- State Slots 全局状态维护

#### 开发工具
- 完整测试套件（pytest + Vitest）
- 代码质量检查（flake8 + ESLint）
- 类型检查（TypeScript + mypy）
- 代码格式化（black + prettier）
- MSW API Mock

### 📝 文档

#### 核心文档
- ✅ README.md - 项目主文档
- ✅ QUICKSTART.md - 5 分钟快速开始
- ✅ INSTALLATION.md - 详细安装部署指南
- ✅ API_REFERENCE.md - 完整 API 参考
- ✅ FAQ.md - 常见问题解答
- ✅ CONTRIBUTING.md - 贡献指南

#### 技术文档
- ✅ 产品需求文档（PRD）
- ✅ 技术调研报告（23 篇技术深潜）
- ✅ 系统架构文档
- ✅ 开发命令参考
- ✅ 测试规范
- ✅ 开发规范

#### 子需求文档
- ✅ 01 模型骨架与插拔 LLM
- ✅ 02 多模态 Token 化与流式引擎
- ✅ 03 实时多模态 RAG 与幻觉控制
- ✅ 04 训练管线与全双工前端

### 🐛 Bug 修复

- 修复 JWT Token 过期处理逻辑
- 修复 RAG 检索结果排序问题
- 修复 WebSocket 连接稳定性
- 修复数据库事务隔离问题
- 修复前端状态同步问题

### ⚡ 性能改进

- 推理速度优化至 85+ tokens/s（Qwen3-8B）
- RAG 检索延迟降低至 45ms
- 显存占用优化 30%
- 前端加载速度提升 50%

### 🔒 安全性

- JWT Token 刷新机制
- CORS 跨域安全配置
- SQL 注入防护
- XSS 攻击防护
- 敏感信息加密存储

---

## [0.2.0] - 2026-03-15

### 新增
- 🎨 前端训练监控页面
- 📊 模型性能指标展示
- 🔍 文档搜索功能
- 📝 对话导出功能（PDF/Markdown）

### 改进
- ⚡ 前端构建速度提升
- 🎯 类型定义完善
- 📱 响应式布局优化

### 修复
- 🐛 修复文档上传进度显示问题
- 🐛 修复对话历史加载问题

---

## [0.1.0] - 2026-02-20

### 新增
- 🎉 项目初始化
- 🏗️ 基础架构搭建
- 📦 Docker 配置
- 🧪 测试框架建立

---

## 版本说明

### 版本号规则

- **主版本号（Major）**: 不兼容的 API 变更
- **次版本号（Minor）**: 向后兼容的功能新增
- **修订号（Patch）**: 向后兼容的问题修复

### 标识说明

- **[未发布]**: 开发中功能
- **新增**: 新功能
- **改进**: 功能优化
- **修复**: Bug 修复
- **安全**: 安全相关更新
- **性能**: 性能优化
- **文档**: 文档更新

---

## 时间线

### Phase 1 (MVP) - 2026 Q1 ✅
- [x] Tri-Transformer 文本模态验证
- [x] 基础 RAG 对话功能
- [x] 前后端基础框架
- [x] 开发文档完善

### Phase 2 (Audio-Text) - 2026 Q2 🚧
- [ ] 音频 Token 化（EnCodec/SNAC）
- [ ] 实时语音对话
- [ ] 低延迟优化（< 500ms）
- [ ] 语音 RAG 检索

### Phase 3 (Omni-Modal) - 2026 Q3 📋
- [ ] 视频 Token 化（VQ-GAN）
- [ ] WebRTC 全双工通信
- [ ] 多模态数字人
- [ ] 企业级部署方案

---

## 贡献者

感谢所有为 Tri-Transformer 做出贡献的开发者！

详见：[GitHub Contributors](https://github.com/your-org/tri-transformer/graphs/contributors)

---

## 相关链接

- 📖 [产品需求文档](./Tri-Transformer 可控对话与 RAG 知识库增强系统.md)
- 🔬 [技术调研报告](./技术调研报告.md)
- 🏗️ [系统架构](./agent/architecture.md)
- 🚀 [快速开始](./QUICKSTART.md)
- 📚 [API 参考](./API_REFERENCE.md)
- 🤝 [贡献指南](../CONTRIBUTING.md)

---

**最后更新**: 2026-04-03  
**维护者**: Tri-Transformer Team
