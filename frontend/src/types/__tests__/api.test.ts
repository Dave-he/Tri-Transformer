import { describe, it, expectTypeOf } from 'vitest';
import type { User, Conversation, Message, Document, MessageSource } from '../api';
import type { WebRTCConnectionState, RTCSignalOffer, RTCSignalAnswer } from '../webrtc';
import type { TrainingConfig, TrainingJob, AvailableModel } from '../trainingConfig';

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

describe('WebRTC Types', () => {
  it('WebRTCConnectionState is valid union type', () => {
    const state: WebRTCConnectionState = 'idle';
    expectTypeOf(state).toBeString();
  });

  it('RTCSignalOffer has sdp and type fields', () => {
    const offer: RTCSignalOffer = { sdp: 'mock_sdp', type: 'offer' };
    expectTypeOf(offer.sdp).toBeString();
    expectTypeOf(offer.type).toEqualTypeOf<'offer'>();
  });

  it('RTCSignalAnswer has sdp and type fields', () => {
    const answer: RTCSignalAnswer = { sdp: 'answer_sdp', type: 'answer' };
    expectTypeOf(answer.sdp).toBeString();
    expectTypeOf(answer.type).toEqualTypeOf<'answer'>();
  });
});

describe('TrainingConfig Types', () => {
  it('TrainingConfig has all required fields', () => {
    const config: TrainingConfig = {
      i_model_id: 'Qwen/Qwen2-Audio-7B',
      o_model_id: 'meta-llama/Llama-3-8B',
      learning_rate: 1e-4,
      batch_size: 8,
      max_steps: 1000,
      phase: 0,
    };
    expectTypeOf(config.phase).toEqualTypeOf<0 | 1 | 2>();
    expectTypeOf(config.learning_rate).toBeNumber();
  });

  it('TrainingJob has jobId field', () => {
    const job: TrainingJob = { jobId: 'job-123' };
    expectTypeOf(job.jobId).toBeString();
  });

  it('AvailableModel has id, name and type', () => {
    const model: AvailableModel = { id: 'qwen2-audio', name: 'Qwen2-Audio', type: 'input' };
    expectTypeOf(model.type).toEqualTypeOf<'input' | 'output'>();
  });
});
