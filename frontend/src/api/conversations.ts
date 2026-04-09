import apiClient from './client';
import type { Conversation, Message, MessageSource } from '@/types/api';

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

export const getConversationsApi = async (): Promise<{ conversations: Conversation[] }> => {
  const { data } = await apiClient.get<{ conversations: Conversation[] }>('/chat/sessions');
  return data;
};

export const createConversationApi = async (title?: string): Promise<Conversation> => {
  const { data } = await apiClient.post<Conversation>('/chat/sessions', { title });
  return data;
};

export const getMessagesApi = async (conversationId: string): Promise<{ messages: Message[] }> => {
  const { data } = await apiClient.get<{ messages: RawMessage[] }>(`/chat/sessions/${conversationId}/history`);
  return { messages: data.messages.map(normalizeMessage) };
};

export const sendMessageApi = async (conversationId: string, content: string): Promise<{ message: Message }> => {
  const { data } = await apiClient.post<{ message: RawMessage }>(`/chat/sessions/${conversationId}/messages`, { content });
  return { message: normalizeMessage(data.message) };
};
