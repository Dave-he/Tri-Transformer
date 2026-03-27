import { create } from 'zustand';
import {
  getConversationsApi,
  createConversationApi,
  getMessagesApi,
  sendMessageApi,
} from '@/api/conversations';
import type { ConversationState } from '@/types/store';
import type { Message } from '@/types/api';

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
      createdAt: new Date().toISOString(),
    };

    set((state) => ({ messages: [...state.messages, userMsg], loading: true, error: null }));

    try {
      const { message } = await sendMessageApi(currentConversationId, content);
      set((state) => ({ messages: [...state.messages, message], loading: false }));
    } catch {
      set({ loading: false, error: '发送消息失败' });
      throw new Error('发送消息失败');
    }
  },
}));
