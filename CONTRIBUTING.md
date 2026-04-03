# 贡献指南

感谢您对 Tri-Transformer 项目的关注！本指南将帮助您了解如何参与项目贡献。

---

## 📑 目录

- [行为准则](#行为准则)
- [贡献方式](#贡献方式)
- [开发环境搭建](#开发环境搭建)
- [代码提交流程](#代码提交流程)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [文档贡献](#文档贡献)
- [PR 审核流程](#pr-审核流程)
- [常见问题](#常见问题)

---

## 行为准则

本项目采用 **贡献者公约** 作为行为准则。我们致力于提供友好、安全、包容的贡献环境。

### 我们的承诺

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

### 不可接受的行为

- 使用性化的语言或图像
- 人身攻击或侮辱性评论
- 公开或私下骚扰
- 未经许可发布他人信息
- 其他不道德或不专业的行为

---

## 贡献方式

### 1. 报告 Bug

发现 Bug？请创建 Issue 并提供：

- 清晰的标题和描述
- 复现步骤
- 期望行为与实际行为
- 环境信息（OS、Python 版本、依赖版本等）
- 相关日志或截图

### 2. 提出新功能

有新想法？请创建 Issue 并说明：

- 功能描述和使用场景
- 为什么这个功能很重要
- 可能的实现方案
- 与其他功能的兼容性

### 3. 提交代码

- Fork 项目仓库
- 创建特性分支
- 实现功能或修复
- 编写测试
- 提交 Pull Request

### 4. 改进文档

- 修复拼写错误
- 补充缺失说明
- 添加示例代码
- 改进翻译质量

### 5. 社区帮助

- 回答 Issue 中的问题
- 分享使用经验
- 推广项目

---

## 开发环境搭建

### 1. Fork 并克隆仓库

```bash
# GitHub 上 Fork 项目

# 克隆到本地
git clone https://github.com/YOUR_USERNAME/tri-transformer.git
cd tri-transformer

# 添加上游仓库
git remote add upstream https://github.com/your-org/tri-transformer.git
```

### 2. 安装依赖

#### 后端

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 如存在
```

#### 前端

```bash
cd frontend

# 安装依赖
pnpm install

# 安装开发工具
pnpm add -D @types/node vitest @vitest/ui
```

### 3. 配置环境

```bash
# 后端
cd backend
cp .env.example .env
# 编辑 .env 配置

# 前端
cd frontend
cp .env.example .env.local
# 编辑 .env.local 配置
```

### 4. 运行测试

```bash
# 后端测试
cd backend
pytest

# 前端测试
cd frontend
pnpm test

# 代码质量检查
cd backend
flake8 app/ tests/
black --check app/ tests/

cd frontend
pnpm lint
pnpm typecheck
```

---

## 代码提交流程

### 1. 创建分支

```bash
# 同步主分支
git checkout main
git pull upstream main

# 创建特性分支
git checkout -b feature/your-feature-name

# 或修复分支
git checkout -b fix/issue-123
```

### 2. 分支命名规范

- **功能**: `feature/description`
- **修复**: `fix/description`
- **文档**: `docs/description`
- **重构**: `refactor/description`
- **测试**: `test/description`
- **性能**: `perf/description`

### 3. 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

#### Type 类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行）
- `refactor`: 重构（既不是新功能也不是修复）
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具变动
- `ci`: CI 配置
- `revert`: 回退提交

#### 示例

```bash
# 新功能
git commit -m "feat(rag): 添加多模态文档检索支持"

# Bug 修复
git commit -m "fix(api): 修复 JWT Token 过期处理逻辑"

# 文档
git commit -m "docs(readme): 更新快速开始指南"

# 重构
git commit -m "refactor(model): 优化 I-Transformer 前向传播逻辑"
```

### 4. 推送分支

```bash
# 推送到远程
git push origin feature/your-feature-name

# 如果是第一次推送
git push --set-upstream origin feature/your-feature-name
```

### 5. 创建 Pull Request

1. 访问 GitHub 仓库
2. 点击 "Pull requests" → "New pull request"
3. 选择您的分支
4. 填写 PR 描述
5. 提交 PR

---

## 代码规范

### Python 代码规范

#### 1. 代码风格

- 遵循 [PEP 8](https://pep8.org/)
- 行宽限制：100 字符
- 使用 4 空格缩进
- 使用 Black 格式化

```bash
# 格式化代码
black app/ tests/

# 检查格式
black --check app/ tests/
```

#### 2. 类型注解

所有公共 API 必须包含类型注解：

```python
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class ChatMessage(BaseModel):
    content: str
    role: str
    sources: Optional[List[Dict[str, Any]]] = None

async def get_message(message_id: str) -> Optional[ChatMessage]:
    """获取消息"""
    ...
```

#### 3. 文档字符串

所有公共函数和类必须有文档字符串：

```python
class RAGRetriever:
    """RAG 检索器，支持向量检索和 BM25 重排序"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        use_rerank: bool = True
    ):
        """
        初始化检索器
        
        Args:
            vector_store: 向量存储
            top_k: 返回结果数
            use_rerank: 是否启用重排序
        """
        ...
    
    async def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        执行检索
        
        Args:
            query: 查询文本
            filter_dict: 过滤条件
            
        Returns:
            检索结果列表
            
        Raises:
            RetrievalError: 检索失败时
        """
        ...
```

#### 4. 异常处理

```python
from app.exceptions import RAGError, RetrievalError

async def safe_retrieve(query: str) -> List[SearchResult]:
    try:
        results = await retriever.retrieve(query)
        return results
    except ConnectionError as e:
        logger.error(f"Vector store connection failed: {e}")
        raise RetrievalError("检索服务不可用") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise RAGError("检索过程发生错误") from e
```

### TypeScript 代码规范

#### 1. 代码风格

- 使用 Prettier 格式化
- 遵循 ESLint 规则
- 使用 2 空格缩进

```bash
# 格式化
pnpm lint:fix

# 检查
pnpm lint
```

#### 2. 类型系统

- 禁止使用 `any`，使用 `unknown` 替代
- 接口使用 `interface`，不用 `type`
-  Props 必须定义接口

```typescript
// ✅ 好的做法
interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  sources?: Source[];
}

interface ChatProps {
  messages: ChatMessage[];
  onSend: (content: string) => Promise<void>;
  isLoading?: boolean;
}

export const Chat: React.FC<ChatProps> = ({ messages, onSend, isLoading }) => {
  // ...
};

// ❌ 避免
type ChatProps = {
  messages: any[];
  onSend: (content: any) => any;
};
```

#### 3. 组件规范

```typescript
import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui';

interface Props {
  initialCount?: number;
  onCountChange?: (count: number) => void;
}

export const Counter: React.FC<Props> = ({
  initialCount = 0,
  onCountChange
}) => {
  const [count, setCount] = useState(initialCount);

  const handleIncrement = useCallback(() => {
    const newCount = count + 1;
    setCount(newCount);
    onCountChange?.(newCount);
  }, [count, onCountChange]);

  return (
    <div>
      <p>Count: {count}</p>
      <Button onClick={handleIncrement}>Increment</Button>
    </div>
  );
};
```

---

## 测试要求

### 后端测试（pytest）

#### 1. 测试文件结构

```python
# tests/test_chat.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def test_create_session(auth_headers: dict):
    """测试创建对话会话"""
    response = client.post(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        json={"title": "Test", "mode": "rag"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test"
    assert "id" in data

async def test_send_message(auth_headers: dict):
    """测试发送消息"""
    response = client.post(
        "/api/v1/chat/sessions/SESSION_ID/messages",
        headers=auth_headers,
        json={"content": "Hello"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert len(data["content"]) > 0
```

#### 2. 测试覆盖率

```bash
# 运行测试并生成覆盖率报告
pytest --cov=app --cov-report=html

# 查看覆盖率
open htmlcov/index.html
```

目标覆盖率：**80%+**

### 前端测试（Vitest）

```typescript
// src/components/chat/__tests__/ChatInput.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatInput } from '../ChatInput';

describe('ChatInput', () => {
  it('renders input field', () => {
    render(<ChatInput onSend={() => {}} />);
    expect(screen.getByPlaceholderText(/type a message/i)).toBeInTheDocument();
  });

  it('calls onSend when submit', async () => {
    const handleSend = vi.fn();
    render(<ChatInput onSend={handleSend} />);
    
    const input = screen.getByPlaceholderText(/type a message/i);
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.submit(input);
    
    expect(handleSend).toHaveBeenCalledWith('Hello');
  });
});
```

---

## 文档贡献

### 1. 文档结构

```
docs/
├── README.md              # 文档索引
├── QUICKSTART.md          # 快速开始
├── INSTALLATION.md        # 安装指南
├── API_REFERENCE.md       # API 文档
├── FAQ.md                 # 常见问题
└── agent/                 # 开发文档
    ├── architecture.md
    ├── conventions.md
    └── testing.md
```

### 2. 文档格式

- 使用 Markdown 格式
- 代码块指定语言
- 添加适当的标题层级
- 包含示例代码

```markdown
# 标题

## 子标题

### 小节

文字说明...

```python
# 代码示例
def example():
    pass
```

**注意**: 重要提示
```

### 3. 文档更新

- 新功能必须附带文档
- Bug 修复更新相关说明
- API 变更更新 API 文档
- 保持文档与代码同步

---

## PR 审核流程

### 1. PR 模板

创建 PR 时，请填写以下信息：

```markdown
## 描述
简要描述此 PR 的目的

## 相关 Issue
Fixes #123

## 变更类型
- [ ] 新功能
- [ ] Bug 修复
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化

## 测试
- [ ] 已添加测试
- [ ] 所有测试通过
- [ ] 已手动测试

## 截图（如适用）
添加前后对比截图

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 已添加文档
- [ ] 无破坏性变更
- [ ] 已更新 CHANGELOG
```

### 2. 审核标准

- **代码质量**: 遵循规范，无 lint 错误
- **功能正确**: 实现需求，测试通过
- **性能影响**: 无明显性能下降
- **文档完整**: 更新相关文档
- **向后兼容**: 无破坏性变更（或明确标注）

### 3. 审核流程

1. **自动检查**: CI 运行测试、lint、类型检查
2. **维护者审核**: 至少 1 名维护者审核
3. **修改反馈**: 根据审核意见修改
4. **合并**: 审核通过后合并到主分支

### 4. 审核时间

- 工作日：24-48 小时
- 周末/节假日：3-5 天

---

## 常见问题

### Q: 我可以同时贡献多个功能吗？

**A**: 建议一次专注于一个功能，每个功能单独的分支和 PR。

### Q: 我的 PR 多久能被审核？

**A**: 通常 1-2 个工作日，如超过 3 天可在 Issue 中 @维护者。

### Q: 如何成为项目维护者？

**A**: 持续贡献高质量代码，积极参与社区，现有维护者会邀请活跃贡献者。

### Q: 代码许可是什么？

**A**: Apache 2.0 许可证，详见 [LICENSE](../LICENSE)。

### Q: 贡献代码需要签署 CLA 吗？

**A**: 目前不需要，但请确保您有权提交该代码。

---

## 致谢

感谢所有为 Tri-Transformer 做出贡献的开发者！

您的每一次贡献都让项目变得更好。

---

**最后更新**: 2026-04-03  
**维护者**: Tri-Transformer Team
