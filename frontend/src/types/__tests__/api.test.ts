import { describe, it, expectTypeOf } from 'vitest';
import type { User, Conversation, Message, Document, MessageSource } from '../api';

describe('API Types', () => {
  it('User type has required fields', () => {
    const user: User = { id: '1', username: 'test', email: 'test@test.com' };
    expectTypeOf(user.id).toBeString();
    expectTypeOf(user.username).toBeString();
    expectTypeOf(user.email).toBeString();
  });

  it('Conversation type has required fields', () => {
    const conv: Conversation = { id: '1', title: 'Test', createdAt: '2026-01-01', updatedAt: '2026-01-01' };
    expectTypeOf(conv.id).toBeString();
    expectTypeOf(conv.title).toBeString();
  });

  it('Message type has role and content', () => {
    const msg: Message = {
      id: '1',
      role: 'user',
      content: 'Hello',
      sources: [],
      createdAt: '2026-01-01',
    };
    expectTypeOf(msg.role).toEqualTypeOf<'user' | 'assistant'>();
    expectTypeOf(msg.sources).toBeArray();
  });

  it('Document type has status field', () => {
    const doc: Document = {
      id: '1',
      name: 'test.pdf',
      type: 'pdf',
      size: 1024,
      status: 'ready',
      createdAt: '2026-01-01',
    };
    expectTypeOf(doc.status).toEqualTypeOf<'processing' | 'ready' | 'failed'>();
  });

  it('MessageSource type has document and chunk', () => {
    const src: MessageSource = { document: 'test.pdf', chunk: 'content...', score: 0.95 };
    expectTypeOf(src.score).toBeNumber();
  });
});
