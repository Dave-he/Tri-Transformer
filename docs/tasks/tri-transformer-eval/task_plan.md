# 任务计划: Tri-Transformer 目标函数与验证体系

## 目标
为 Tri-Transformer RAG 系统设计完整的目标函数体系、Agent 自主 Ground Truth 发现引擎和自动化 Evaluation Pipeline，使系统能在无人工介入的情况下自我评估和验证。

## 当前阶段
Phase 2: 技术方案设计

## 阶段规划

### Phase 1: 需求与发现
- [x] 读取并解析 PRD 文档
- [x] 识别三大核心模块（目标函数/GT构建/Eval框架）
- [x] 生成 requirement.yaml + requirement.md
- **Status:** complete

### Phase 2: 技术方案设计（Stage 3）
- [ ] 设计 RAG 检索质量损失函数（Retrieval Loss）架构
- [ ] 设计 C 分支控制对齐损失函数架构
- [ ] 设计幻觉抑制损失函数架构
- [ ] 设计 Agent Ground Truth 构建引擎流程
- [ ] 设计 Evaluation Pipeline 架构
- [ ] 生成 tech-solution.yaml + tech-solution.md
- **Status:** in_progress

### Phase 3: 任务拆解（Stage 5）
- [ ] 将技术方案拆解为 TDD 任务清单
- [ ] 确保每个任务都有对应的测试用例
- [ ] 生成 plan.yaml + plan.md
- **Status:** pending

### Phase 4: 方案验证（Stage 6）
- [ ] 验证技术方案与任务清单的一致性
- [ ] 确认 CI Gate 指标可量化
- [ ] 生成 verification-report.md
- **Status:** pending

### Phase 5: TDD 实现（Stage 7）
- [ ] 实现各目标函数 Python 模块
- [ ] 实现 Ground Truth 构建引擎
- [ ] 实现 Evaluation Pipeline
- [ ] 所有测试通过（RED→GREEN）
- **Status:** pending

### Phase 6: 验收
- [ ] 运行完整 evaluation pipeline
- [ ] 确认 CI Gate 指标达标
- [ ] 文档完善
- **Status:** pending

## 关键问题
1. Agent GT 发现：如何从文档自动提取高质量问答对？→ 采用 LLM 生成 + 双模型交叉验证
2. 控制对齐损失：如何量化 C 分支控制信号质量？→ 对比学习 + NLI 分类
3. 离线运行约束：所有评估组件必须本地可运行 → 使用开源模型（BGE, DeBERTa-NLI）

## 决策记录
| 决策 | 理由 |
|------|------|
| 使用 RAGAS 框架 | 业界标准 RAG 评估工具，无商业依赖 |
| BGE Reranker 作为质量评分器 | 与系统内嵌入模型一致，无额外依赖 |
| DeBERTa-NLI 作为幻觉检测器 | 开源高质量 NLI 模型，本地可运行 |
| JSON Lines + HF Dataset 双格式 | 兼容 RAGAS 和 HuggingFace evaluate |

## 错误记录
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------| 
| - | - | - |

## 备注
- 所有组件必须支持 CPU-only 模式
- Docker Compose 一键部署
- 与 CI/CD 集成通过 GitHub Actions
