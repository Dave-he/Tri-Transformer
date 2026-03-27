import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MessageBubble } from '../MessageBubble';

describe('MessageBubble', () => {
  it('renders user message content', () => {
    render(
      <MessageBubble
        message={{
          id: '1',
          role: 'user',
          content: 'Hello, what is RAG?',
          sources: [],
          createdAt: '2026-01-01T00:00:00Z',
        }}
      />
    );
    expect(screen.getByText('Hello, what is RAG?')).toBeDefined();
  });

  it('renders assistant message content', () => {
    render(
      <MessageBubble
        message={{
          id: '2',
          role: 'assistant',
          content: 'RAG stands for Retrieval-Augmented Generation.',
          sources: [],
          createdAt: '2026-01-01T00:00:00Z',
        }}
      />
    );
    expect(screen.getByText('RAG stands for Retrieval-Augmented Generation.')).toBeDefined();
  });

  it('does not render source panel when sources is empty', () => {
    render(
      <MessageBubble
        message={{
          id: '3',
          role: 'assistant',
          content: 'Answer',
          sources: [],
          createdAt: '2026-01-01T00:00:00Z',
        }}
      />
    );
    expect(screen.queryByText(/来源/)).toBeNull();
  });

  it('renders source panel when sources is not empty', () => {
    render(
      <MessageBubble
        message={{
          id: '4',
          role: 'assistant',
          content: 'Answer with sources',
          sources: [{ document: 'rag.pdf', chunk: 'RAG content here', score: 0.95 }],
          createdAt: '2026-01-01T00:00:00Z',
        }}
      />
    );
    expect(screen.getByText(/知识来源/)).toBeDefined();
  });
});
