# AGENTS.md

This file provides guidance to codeflicker when working with code in this repository.

## WHY: Purpose and Goals

Tri-Transformer 是基于三分支 Transformer 架构的 AI 幻觉检测与实时通信系统，提供高精度幻觉检测（PyTorch I/C/O 三分支模型）、RAG 知识库问答、WebSocket 实时通信，以及 React 数据可视化前端。

## WHAT: Technical Stack

- **Frontend**: React 18 + TypeScript + Vite, Ant Design, Zustand, Recharts, Axios
- **Backend**: FastAPI (Python 3.10), SQLAlchemy async, Pydantic v2, JWT auth
- **ML Model**: PyTorch Tri-Transformer (ITransformer / CTransformer / OTransformer)
- **RAG**: ChromaDB + sentence-transformers + BM25 reranking
- **Eval**: Custom hallucination/RAG/control-alignment loss functions
- **Infra**: Docker + docker-compose, Milvus vector DB

## HOW: Core Development Workflow

```bash
# Frontend
cd frontend && pnpm dev          # dev server (port 3000)
cd frontend && pnpm test         # Vitest tests
cd frontend && pnpm lint && pnpm typecheck

# Backend
cd backend && uvicorn app.main:app --reload  # dev server (port 8000)
cd backend && pytest && flake8 app/ tests/

# Docker
docker-compose up -d
```

## Progressive Disclosure

For detailed information, consult these documents as needed:

- `docs/agent/development_commands.md` - All build, test, lint, release commands
- `docs/agent/architecture.md` - Module structure and architectural patterns
- `docs/agent/testing.md` - Test setup, frameworks, and conventions
- `docs/agent/conventions.md` - Code style and directory conventions

**When working on a task, first determine which documentation is relevant, then read only those files.**
