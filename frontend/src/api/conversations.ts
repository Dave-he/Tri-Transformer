import apiClient from './client';
import type { Conversation, Message } from '@/types/api';

export const getConversationsApi = async (): Promise<{ conversations: Conversation[] }> => {
  const { data } = await apiClient.get<{ conversations: Conversation[] }>('/conversations');
  return data;
};

export const createConversationApi = async (title?: string): Promise<Conversation> => {
  const { data } = await apiClient.post<Conversation>('/conversations', { title });
  return data;
};

export const getMessagesApi = async (conversationId: string): Promise<{ messages: Message[] }> => {
  const { data } = await apiClient.get<{ messages: Message[] }>(`/conversations/${conversationId}/messages`);
  return data;
};

export const sendMessageApi = async (conversationId: string, content: string): Promise<{ message: Message }> => {
  const { data } = await apiClient.post<{ message: Message }>(`/conversations/${conversationId}/messages`, { content });
  return data;
};
