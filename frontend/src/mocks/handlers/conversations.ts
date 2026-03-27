import { http, HttpResponse } from 'msw';

const mockConversations = [
  { id: 'conv-1', title: '关于 RAG 的探讨', createdAt: '2026-01-01T00:00:00Z', updatedAt: '2026-01-01T01:00:00Z' },
  { id: 'conv-2', title: 'Transformer 架构问题', createdAt: '2026-01-02T00:00:00Z', updatedAt: '2026-01-02T01:00:00Z' },
];

export const conversationHandlers = [
  http.get('http://localhost:8000/api/v1/conversations', () => {
    return HttpResponse.json({ conversations: mockConversations });
  }),

  http.post('http://localhost:8000/api/v1/conversations', async ({ request }) => {
    const body = await request.json() as { title?: string };
    return HttpResponse.json({
      id: `conv-${Date.now()}`,
      title: body.title ?? '新对话',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }, { status: 201 });
  }),

  http.get('http://localhost:8000/api/v1/conversations/:id/messages', ({ params }) => {
    return HttpResponse.json({
      messages: [
        {
          id: 'msg-1',
          role: 'user',
          content: '什么是 RAG？',
          sources: [],
          createdAt: '2026-01-01T00:00:00Z',
        },
        {
          id: 'msg-2',
          role: 'assistant',
          content: `RAG（检索增强生成）是一种将检索系统与生成模型结合的技术架构，对话ID: ${params['id']}`,
          sources: [{ document: 'rag-overview.pdf', chunk: 'RAG 技术简介...', score: 0.95 }],
          createdAt: '2026-01-01T00:01:00Z',
        },
      ],
    });
  }),

  http.post('http://localhost:8000/api/v1/conversations/:id/messages', async ({ request }) => {
    const body = await request.json() as { content: string };
    return HttpResponse.json({
      message: {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: `基于知识库的回答：${body.content} 相关内容已从知识库检索。`,
        sources: [
          { document: 'knowledge-base.pdf', chunk: '相关段落内容...', score: 0.92 },
          { document: 'technical-docs.md', chunk: '技术文档摘录...', score: 0.87 },
        ],
        createdAt: new Date().toISOString(),
      },
    });
  }),
];
