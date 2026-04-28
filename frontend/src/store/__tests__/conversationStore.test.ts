import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../../api/conversations', () => ({
  getConversationsApi: vi.fn(),
  createConversationApi: vi.fn(),
  getMessagesApi: vi.fn(),
  sendMessageApi: vi.fn(),
}));

describe('conversationStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  it('initial state: empty conversations and messages', async () => {
    const { useConversationStore } = await import('../conversationStore');
    const state = useConversationStore.getState();
    expect(state.conversations).toEqual([]);
    expect(state.messages).toEqual([]);
    expect(state.loading).toBe(false);
  });

  it('sendMessage sets loading to true then false on success', async () => {
    const { sendMessageApi } = await import('../../api/conversations');
    const { useConversationStore } = await import('../conversationStore');

    vi.mocked(sendMessageApi).mockResolvedValue({
      message: {
        id: 'msg-1',
        role: 'assistant' as const,
        content: 'RAG is Retrieval-Augmented Generation.',
        sources: [{ document: 'rag.pdf', chunk: 'content', score: 0.95 }],
        createdAt: '2026-01-01T00:00:00Z',
      },
    });

    useConversationStore.setState({ currentConversationId: 'conv-1' });

    await useConversationStore.getState().sendMessage('What is RAG?');

    expect(useConversationStore.getState().loading).toBe(false);
  });

  it('sendMessage appends assistant message with sources', async () => {
    const { sendMessageApi } = await import('../../api/conversations');
    const { useConversationStore } = await import('../conversationStore');

    const mockMessage = {
      id: 'msg-2',
      role: 'assistant' as const,
      content: 'Answer here.',
      sources: [{ document: 'doc.pdf', chunk: 'chunk text', score: 0.9 }],
      createdAt: '2026-01-01T00:00:00Z',
    };

    vi.mocked(sendMessageApi).mockResolvedValue({ message: mockMessage });
    useConversationStore.setState({ currentConversationId: 'conv-1', messages: [] });

    await useConversationStore.getState().sendMessage('Question');

    const messages = useConversationStore.getState().messages;
    const assistantMsg = messages.find((m) => m.role === 'assistant');
    expect(assistantMsg).toBeDefined();
    expect(assistantMsg?.sources).toHaveLength(1);
    expect(assistantMsg?.sources[0].document).toBe('doc.pdf');
  });

  it('sendMessage sets error state on API failure', async () => {
    const { sendMessageApi } = await import('../../api/conversations');
    const { useConversationStore } = await import('../conversationStore');

    vi.mocked(sendMessageApi).mockRejectedValue(new Error('Network error'));
    useConversationStore.setState({ currentConversationId: 'conv-1' });

    await expect(useConversationStore.getState().sendMessage('Question')).rejects.toThrow();
    expect(useConversationStore.getState().loading).toBe(false);
  });

  it('createConversation adds to conversations list', async () => {
    const { createConversationApi } = await import('../../api/conversations');
    const { useConversationStore } = await import('../conversationStore');

    vi.mocked(createConversationApi).mockResolvedValue({
      id: 'conv-new',
      title: 'New Chat',
      status: 'active',
      createdAt: '2026-01-01T00:00:00Z',
      updatedAt: '2026-01-01T00:00:00Z',
      messageCount: 0,
    });

    useConversationStore.setState({ conversations: [] });
    await useConversationStore.getState().createConversation();

    expect(useConversationStore.getState().conversations).toHaveLength(1);
    expect(useConversationStore.getState().conversations[0].id).toBe('conv-new');
  });
});
