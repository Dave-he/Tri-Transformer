# AI Flow 工作流说明

## 需求到代码工作流

```
1. 需求质量门禁（requirement-quality-gate）
   ↓ PASS
2. 技术方案（tech-solution）
   → docs/tasks/{task-id}/tech-solution.yaml
   ↓
3. 任务拆解（plan-from-tech-solution）
   → docs/tasks/{task-id}/plan.yaml
   → docs/tasks/{task-id}/plan.md
   ↓
4. 方案验证（verify-from-tech-solution）
   ↓ PASS
5. TDD 实现（iterate-from-plan-and-tests）
   ↓
6. 文件变更审查 + 代码审查
```

## 常用命令

- `"完成需求 T123456"` - 启动 ai-flow 全流程
- `"出技术方案"` - 仅生成技术方案
- `"拆分任务"` - 仅生成任务清单
- `"验证方案"` - 仅校验方案一致性

## 文档结构

```
docs/
├── agent/              # AI Flow 约定（本目录）
│   ├── conventions.md
│   ├── architecture.md
│   ├── development_commands.md
│   └── requirement-template.md
├── tasks/              # 任务文档
│   └── {task-id}/
│       ├── tech-solution.yaml
│       ├── plan.yaml
│       └── plan.md
└── research/           # 研发资产
    ├── rd-assets.md
    └── snippets-index.md
```
