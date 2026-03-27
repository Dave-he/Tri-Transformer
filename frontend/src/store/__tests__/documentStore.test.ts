import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../../api/documents', () => ({
  getDocumentsApi: vi.fn(),
  uploadDocumentApi: vi.fn(),
  deleteDocumentApi: vi.fn(),
  searchDocumentsApi: vi.fn(),
}));

describe('documentStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  it('initial state: empty documents and null uploadProgress', async () => {
    const { useDocumentStore } = await import('../documentStore');
    const state = useDocumentStore.getState();
    expect(state.documents).toEqual([]);
    expect(state.uploadProgress).toBeNull();
    expect(state.searchResults).toEqual([]);
  });

  it('upload updates uploadProgress and adds document to list', async () => {
    const { uploadDocumentApi } = await import('../../api/documents');
    const { useDocumentStore } = await import('../documentStore');

    vi.mocked(uploadDocumentApi).mockResolvedValue({
      id: 'doc-1',
      name: 'test.pdf',
      type: 'pdf',
      size: 1024,
      status: 'processing' as const,
      createdAt: '2026-01-01T00:00:00Z',
    });

    const file = new File(['content'], 'test.pdf', { type: 'application/pdf' });
    await useDocumentStore.getState().upload(file);

    expect(useDocumentStore.getState().uploadProgress).toBeNull();
    expect(useDocumentStore.getState().documents).toHaveLength(1);
  });

  it('delete removes document from list', async () => {
    const { deleteDocumentApi } = await import('../../api/documents');
    const { useDocumentStore } = await import('../documentStore');

    vi.mocked(deleteDocumentApi).mockResolvedValue({ message: 'deleted' });
    useDocumentStore.setState({
      documents: [
        { id: 'doc-1', name: 'test.pdf', type: 'pdf', size: 1024, status: 'ready', createdAt: '2026-01-01T00:00:00Z' },
      ],
    });

    await useDocumentStore.getState().deleteDocument('doc-1');

    expect(useDocumentStore.getState().documents).toHaveLength(0);
  });

  it('search returns searchResults', async () => {
    const { searchDocumentsApi } = await import('../../api/documents');
    const { useDocumentStore } = await import('../documentStore');

    vi.mocked(searchDocumentsApi).mockResolvedValue({
      results: [{ document: 'test.pdf', chunk: 'relevant content', score: 0.92 }],
    });

    await useDocumentStore.getState().search('RAG retrieval', 5);

    expect(useDocumentStore.getState().searchResults).toHaveLength(1);
    expect(useDocumentStore.getState().searchResults[0].score).toBe(0.92);
  });

  it('upload failure sets error state', async () => {
    const { uploadDocumentApi } = await import('../../api/documents');
    const { useDocumentStore } = await import('../documentStore');

    vi.mocked(uploadDocumentApi).mockRejectedValue(new Error('Upload failed'));

    const file = new File(['content'], 'test.pdf', { type: 'application/pdf' });
    await expect(useDocumentStore.getState().upload(file)).rejects.toThrow('Upload failed');
    expect(useDocumentStore.getState().uploadProgress).toBeNull();
  });
});
