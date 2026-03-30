/**
 * 代码片段 - conversationStore
 * 
 * @category stores
 * @tags stores, react, typescript, dependencies
 * @dependencies zustand
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/store/conversationStore.ts
 * 评分: 4.40
 * 复杂度: 2
 */

export const useConversationStore = create<ConversationState>((set, get) => ({
  conversations: [],
  currentConversationId: null,
  messages: [],
  loading: false,
  error: null,

  fetchConversations: async () => {
    set({ loading: true, error: null });
    try {
      const { conversations } = await getConversationsApi();
      set({ conversations, loading: false });
    } catch {
      set({ loading: false, error: '加载对话列表失败' });
    }
  },

  createConversation: async (title?: string) => {
    set({ loading: true, error: null });
    try {
      const conversation = await createConversationApi(title);
      set((state) => ({
        conversations: [conversation, ...state.conversations],
        currentConversationId: conversation.id,
        messages: [],
        loading: false,
      }));
    } catch {
      set({ loading: false, error: '创建对话失败' });
      throw new Error('创建对话失败');
    }
  },

  setActiveConversation: async (id: string) => {
    set({ currentConversationId: id, loading: true, error: null });
    try {
      const { messages } = await getMessagesApi(id);
      set({ messages, loading: false });
    } catch {
      set({ loading: false, error: '加载消息失败' });
    }
  },

  sendMessage: async (content: string) => {
    const { currentConversationId } = get();
    if (!currentConversationId) throw new Error('未选择对话');

    const userMsg: Message = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content,
      sources: [],
      createdAt: new Date(
// ... 更多实现