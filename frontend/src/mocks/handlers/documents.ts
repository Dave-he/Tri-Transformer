import { http, HttpResponse } from 'msw';

const mockDocuments = [
  { id: 'doc-1', name: 'rag-overview.pdf', type: 'pdf', size: 204800, status: 'ready', createdAt: '2026-01-01T00:00:00Z' },
  { id: 'doc-2', name: 'transformer-arch.md', type: 'markdown', size: 51200, status: 'ready', createdAt: '2026-01-02T00:00:00Z' },
];

export const documentHandlers = [
  http.get('http://localhost:8002/api/v1/documents', () => {
    return HttpResponse.json({ documents: mockDocuments });
  }),

  http.post('http://localhost:8002/api/v1/documents/upload', () => {
    return HttpResponse.json({
      id: `doc-${Date.now()}`,
      name: 'uploaded-file.pdf',
      type: 'pdf',
      size: 102400,
      status: 'processing',
      createdAt: new Date().toISOString(),
    }, { status: 201 });
  }),

  http.delete('http://localhost:8002/api/v1/documents/:id', () => {
    return HttpResponse.json({ message: 'Document deleted successfully' });
  }),

  http.post('http://localhost:8002/api/v1/documents/search', async ({ request }) => {
    const body = await request.json() as { query: string; topK?: number };
    const topK = body.topK ?? 10;
    return HttpResponse.json({
      results: Array.from({ length: Math.min(topK, 3) }, (_, i) => ({
        document: mockDocuments[i % mockDocuments.length]?.name ?? 'doc.pdf',
        chunk: `与查询"${body.query}"相关的文档段落 ${i + 1}...`,
        score: 0.95 - i * 0.05,
      })),
    });
  }),
];
