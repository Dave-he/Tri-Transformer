import { describe, it, expect } from 'vitest';
import { exportToJSON, exportToMarkdown } from '../exportConversation';
import type { Message } from '../../types/api';

const mockMessages: Message[] = [
  {
    id: '1',
    role: 'user',
    content: 'What is RAG?',
    sources: [],
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: '2',
    role: 'assistant',
    content: 'RAG is Retrieval-Augmented Generation.',
    sources: [{ document: 'rag.pdf', chunk: 'content', score: 0.9 }],
    createdAt: '2026-01-01T00:01:00Z',
  },
];

describe('exportToJSON', () => {
  it('returns a valid JSON string', () => {
    const result = exportToJSON(mockMessages);
    expect(() => JSON.parse(result)).not.toThrow();
  });

  it('contains all message content', () => {
    const result = exportToJSON(mockMessages);
    const parsed = JSON.parse(result);
    expect(parsed).toHaveLength(2);
    expect(parsed[0].content).toBe('What is RAG?');
    expect(parsed[1].content).toBe('RAG is Retrieval-Augmented Generation.');
  });

  it('handles empty messages array without error', () => {
    expect(() => exportToJSON([])).not.toThrow();
    const result = exportToJSON([]);
    expect(JSON.parse(result)).toEqual([]);
  });
});

describe('exportToMarkdown', () => {
  it('contains message roles', () => {
    const result = exportToMarkdown(mockMessages);
    expect(result).toContain('用户');
    expect(result).toContain('助手');
  });

  it('contains message content', () => {
    const result = exportToMarkdown(mockMessages);
    expect(result).toContain('What is RAG?');
    expect(result).toContain('RAG is Retrieval-Augmented Generation.');
  });

  it('handles empty messages without error', () => {
    expect(() => exportToMarkdown([])).not.toThrow();
  });
});
