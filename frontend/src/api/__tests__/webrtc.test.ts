import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { sendOffer, sendCandidate, sendInterrupt } from '../webrtc';

const server = setupServer(
  http.post('http://localhost:8002/api/v1/webrtc/offer', () => {
    return HttpResponse.json({ sdp: 'answer_sdp', type: 'answer' });
  }),
  http.post('http://localhost:8002/api/v1/webrtc/candidate', () => {
    return HttpResponse.json({ ok: true });
  }),
  http.post('http://localhost:8002/api/v1/webrtc/interrupt', () => {
    return HttpResponse.json({ ok: true });
  })
);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('WebRTC API', () => {
  it('sendOffer sends POST /webrtc/offer and returns SDP answer', async () => {
    const result = await sendOffer({ sdp: 'offer_sdp', type: 'offer' });
    expect(result.sdp).toBe('answer_sdp');
    expect(result.type).toBe('answer');
  });

  it('sendCandidate sends POST /webrtc/candidate', async () => {
    const result = await sendCandidate({ candidate: 'candidate_string', sdpMid: '0', sdpMLineIndex: 0 });
    expect(result.ok).toBe(true);
  });

  it('sendInterrupt sends POST /webrtc/interrupt', async () => {
    const result = await sendInterrupt();
    expect(result.ok).toBe(true);
  });
});
