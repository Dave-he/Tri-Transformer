import { http, HttpResponse } from 'msw';

export const webrtcHandlers = [
  http.post('http://localhost:8002/api/v1/webrtc/offer', () => {
    return HttpResponse.json({ sdp: 'mock_answer_sdp_from_server', type: 'answer' });
  }),

  http.post('http://localhost:8002/api/v1/webrtc/candidate', () => {
    return HttpResponse.json({ ok: true });
  }),

  http.post('http://localhost:8002/api/v1/webrtc/interrupt', () => {
    return HttpResponse.json({ ok: true });
  }),
];
