import type { Message } from '@/types/api';

export const exportToJSON = (messages: Message[]): string => {
  return JSON.stringify(messages, null, 2);
};

export const exportToMarkdown = (messages: Message[]): string => {
  if (messages.length === 0) return '';
  return messages
    .map((m) => {
      const roleLabel = m.role === 'user' ? '用户' : '助手';
      const timestamp = new Date(m.createdAt).toLocaleString();
      return `## ${roleLabel} (${timestamp})\n\n${m.content}`;
    })
    .join('\n\n---\n\n');
};

export const downloadConversation = (messages: Message[], format: 'json' | 'markdown' = 'json'): void => {
  const content = format === 'json' ? exportToJSON(messages) : exportToMarkdown(messages);
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
