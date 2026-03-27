# 任务计划: Tri-Transformer 模型骨架 + PyTorch 训练集成

## 目标
在现有 FastAPI 后端基础上，新增 Tri-Transformer 三分支 PyTorch 模型骨架及训练器，并通过 BackgroundTasks 驱动训练流程，将训练指标写回数据库。

## 当前阶段
Phase 2（技术方案）

## 阶段规划

### Phase 1: 需求与发现
- [x] 理解用户意图：搭建 Tri-Transformer 骨架 + PyTorch 训练服务
- [x] 探索现有代码结构：FastAPI 后端已有 train API、TrainService、TrainJob ORM
- [x] 识别约束：不引入 transformers 库、不破坏现有测试、CPU 可运行
- **Status:** complete

### Phase 2: 技术方案设计
- [x] 定义模型文件结构
- [x] 设计三分支模型接口
- [x] 设计 Trainer 接口
- [x] 设计 BackgroundTask 集成方案
- **Status:** in_progress

### Phase 3: 实现（TDD）
- [ ] T1: 编写 test_tri_transformer.py（RED）
- [ ] T2: 实现 tri_transformer.py（GREEN）
- [ ] T3: 编写 test_pytorch_trainer.py（RED）
- [ ] T4: 实现 pytorch_trainer.py（GREEN）
- [ ] T5: 扩展 train_service.py + 修改 train API（集成 BackgroundTask）
- [ ] T6: 扩展 config.py（新增训练超参数）
- [ ] T7: 更新 requirements.txt（新增 torch）
- **Status:** pending

### Phase 4: 验证
- [ ] 运行 pytest tests/ -v
- [ ] 确认现有测试全部通过
- [ ] 确认新增测试通过
- **Status:** pending

## 关键问题
1. torch 版本如何添加到 requirements.txt？→ 添加 `torch>=2.0.0` CPU 版本
2. BackgroundTask 如何安全更新 DB？→ 使用独立 session（不共享请求 session）
3. cancel 信号如何传递？→ threading.Event + job_id 全局字典管理

## 决策记录
| 决策 | 理由 |
|------|------|
| 纯 torch 不引入 transformers | 轻量依赖，骨架验证场景 |
| threading.Event 取消 | 简单可靠，避免复杂异步信号 |
| 随机 Tensor 模拟数据 | 避免真实数据集依赖，专注骨架验证 |
| d_model=32 测试环境 | 加速测试执行，不依赖 GPU |

## 错误记录
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
|      | 0 | — |
