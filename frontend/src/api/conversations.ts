import apiClient from './client';
import type { Conversation, Message, MessageSource, PaginationInfo } from '@/types/api';

interface RawMessage {
  message_id?: string;
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  sources: Array<{ document?: string; chunk?: string; score?: number }>;
  created_at?: string;
  createdAt?: string;
  hallucination_detected?: boolean;
}

function normalizeMessage(raw: RawMessage): Message {
  return {
    id: raw.message_id ?? raw.id ?? '',
    role: raw.role,
    content: raw.content,
    sources: (raw.sources ?? []).map((s): MessageSource => ({
      document: s.document ?? '',
      chunk: s.chunk ?? '',
      score: s.score ?? 0,
    })),
    createdAt: raw.created_at ?? raw.createdAt ?? '',
    hallucinationDetected: raw.hallucination_detected,
  };
}

export const getConversationsApi = async (page = 1, pageSize = 20): Promise<{ conversations: Conversation[]; pagination: PaginationInfo }> => {
  const { data } = await apiClient.get<{ conversations: Conversation[]; pagination: PaginationInfo }>('/chat/sessions', { params: { page, page_size: pageSize } });
  return data;
};

export const createConversationApi = async (title?: string): Promise<Conversation> => {
  const { data } = await apiClient.post<Conversation>('/chat/sessions', { title });
  return data;
};

export const deleteConversationApi = async (conversationId: string): Promise<{ message: string }> => {
  const { data } = await apiClient.delete<{ message: string }>(`/chat/sessions/${conversationId}`);
  return data;
};

export const getMessagesApi = async (conversationId: string): Promise<{ messages: Message[] }> => {
  const { data } = await apiClient.get<RawMessage[]>(`/chat/sessions/${conversationId}/history`);
  return { messages: data.map(normalizeMessage) };
};

export const sendMessageApi = async (conversationId: string, content: string): Promise<{ message: Message }> => {
  const { data } = await apiClient.post<{ message: RawMessage }>(`/chat/sessions/${conversationId}/messages`, { content });
  return { message: normalizeMessage(data.message) };
};
