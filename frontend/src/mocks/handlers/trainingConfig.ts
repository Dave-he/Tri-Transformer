import { http, HttpResponse } from 'msw';

export const trainingConfigHandlers = [
  http.post('http://localhost:8000/api/v1/training/start', () => {
    return HttpResponse.json({ jobId: 'test-job-1' });
  }),

  http.get('http://localhost:8000/api/v1/training/progress', () => {
    return HttpResponse.json({
      jobId: 'test-job-1',
      phase: 0,
      step: 100,
      maxSteps: 1000,
      loss: 2.5,
      lr: 1e-4,
      status: 'running',
    });
  }),

  http.get('http://localhost:8000/api/v1/models/available', () => {
    return HttpResponse.json({
      models: [
        { id: 'qwen2-audio', name: 'Qwen2-Audio-7B', type: 'input' },
        { id: 'qwen2-vl', name: 'Qwen2-VL-7B', type: 'input' },
        { id: 'llama3-8b', name: 'Llama-3-8B', type: 'output' },
        { id: 'gpt2', name: 'GPT-2', type: 'output' },
      ],
    });
  }),
];
