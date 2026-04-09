import { http, HttpResponse } from 'msw';

const historyPoints = Array.from({ length: 10 }, (_, i) => ({
  timestamp: new Date(Date.now() - (9 - i) * 3600000).toISOString(),
  retrievalAccuracy: 0.85 + i * 0.01,
  bleuScore: 0.7 + i * 0.01,
  hallucinationRate: 0.08 - i * 0.005,
}));

export const trainingHandlers = [
  http.get('http://localhost:8002/api/v1/training/status', () => {
    return HttpResponse.json({
      phase: 'Stage 2: C-Transformer Training',
      progress: 45,
      eta: '约 2 小时后完成',
      message: '正在训练控制中枢分支...',
    });
  }),

  http.get('http://localhost:8002/api/v1/metrics', () => {
    return HttpResponse.json({
      current: {
        retrievalAccuracy: 0.92,
        bleuScore: 0.78,
        hallucinationRate: 0.03,
      },
      history: historyPoints,
    });
  }),
];
