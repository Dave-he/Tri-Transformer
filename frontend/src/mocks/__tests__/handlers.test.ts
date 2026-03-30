import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { authHandlers } from '../handlers/auth';
import { conversationHandlers } from '../handlers/conversations';
import { documentHandlers } from '../handlers/documents';
import { trainingHandlers } from '../handlers/training';
import { webrtcHandlers } from '../handlers/webrtc';
import { trainingConfigHandlers } from '../handlers/trainingConfig';

const server = setupServer(
  ...authHandlers,
  ...conversationHandlers,
  ...documentHandlers,
  ...trainingHandlers,
  ...webrtcHandlers,
  ...trainingConfigHandlers
);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Auth handlers', () => {
  it('POST /api/v1/auth/login returns token and user', async () => {
    const res = await fetch('http://localhost:8000/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'test', password: 'password' }),
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('token');
    expect(data).toHaveProperty('user');
    expect(data.user).toHaveProperty('id');
  });

  it('POST /api/v1/auth/login returns 401 for invalid credentials', async () => {
    const res = await fetch('http://localhost:8000/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'wrong', password: 'wrong' }),
    });
    expect(res.status).toBe(401);
  });
});

describe('Conversation handlers', () => {
  it('GET /api/v1/conversations returns conversation list', async () => {
    const res = await fetch('http://localhost:8000/api/v1/conversations', {
      headers: { Authorization: 'Bearer test-token' },
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('conversations');
    expect(Array.isArray(data.conversations)).toBe(true);
  });

  it('POST /api/v1/conversations/:id/messages returns message with sources', async () => {
    const res = await fetch('http://localhost:8000/api/v1/conversations/conv-1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer test-token' },
      body: JSON.stringify({ content: 'What is RAG?' }),
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('message');
    expect(data.message).toHaveProperty('sources');
    expect(Array.isArray(data.message.sources)).toBe(true);
  });
});

describe('Document handlers', () => {
  it('POST /api/v1/documents/upload returns processing status', async () => {
    const formData = new FormData();
    formData.append('file', new Blob(['content'], { type: 'application/pdf' }), 'test.pdf');
    const res = await fetch('http://localhost:8000/api/v1/documents/upload', {
      method: 'POST',
      headers: { Authorization: 'Bearer test-token' },
      body: formData,
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('id');
    expect(data.status).toBe('processing');
  });
});

describe('Training handlers', () => {
  it('GET /api/v1/metrics returns metrics data', async () => {
    const res = await fetch('http://localhost:8000/api/v1/metrics', {
      headers: { Authorization: 'Bearer test-token' },
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('current');
    expect(data.current).toHaveProperty('retrievalAccuracy');
    expect(data.current).toHaveProperty('bleuScore');
    expect(data.current).toHaveProperty('hallucinationRate');
  });
});

describe('WebRTC handlers', () => {
  it('POST /api/v1/webrtc/offer returns SDP answer', async () => {
    const res = await fetch('http://localhost:8000/api/v1/webrtc/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer test-token' },
      body: JSON.stringify({ sdp: 'offer_sdp', type: 'offer' }),
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('sdp');
    expect(data.type).toBe('answer');
  });

  it('POST /api/v1/webrtc/interrupt returns ok', async () => {
    const res = await fetch('http://localhost:8000/api/v1/webrtc/interrupt', {
      method: 'POST',
      headers: { Authorization: 'Bearer test-token' },
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data.ok).toBe(true);
  });
});

describe('TrainingConfig handlers', () => {
  it('POST /api/v1/training/start returns jobId', async () => {
    const res = await fetch('http://localhost:8000/api/v1/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer test-token' },
      body: JSON.stringify({ i_model_id: 'qwen2', o_model_id: 'llama3', learning_rate: 1e-4, batch_size: 8, max_steps: 100, phase: 0 }),
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('jobId');
  });

  it('GET /api/v1/training/progress returns progress object', async () => {
    const res = await fetch('http://localhost:8000/api/v1/training/progress', {
      headers: { Authorization: 'Bearer test-token' },
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('step');
    expect(data).toHaveProperty('maxSteps');
    expect(data).toHaveProperty('status');
  });

  it('GET /api/v1/models/available returns models array', async () => {
    const res = await fetch('http://localhost:8000/api/v1/models/available', {
      headers: { Authorization: 'Bearer test-token' },
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(Array.isArray(data.models)).toBe(true);
  });
});
