import { useConversationStore } from '@/store/conversationStore';
import type { Message } from '@/types/api';

export const exportConversationToJSON = (messages: Message[]): string => {
  return JSON.stringify(messages, null, 2);
};

export const exportConversationToMarkdown = (messages: Message[]): string => {
  return messages
    .map((m) => `**${m.role === 'user' ? '用户' : '助手'}**\n\n${m.content}`)
    .join('\n\n---\n\n');
};

export const useConversation = () => {
  const store = useConversationStore();

  const exportHistory = (format: 'json' | 'markdown' = 'json') => {
    const { messages } = store;
    const content = format === 'json' ? exportConversationToJSON(messages) : exportConversationToMarkdown(messages);
    const mimeType = format === 'json' ? 'application/json' : 'text/markdown';
    const ext = format === 'json' ? 'json' : 'md';

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation-${Date.now()}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return { ...store, exportHistory };
};
