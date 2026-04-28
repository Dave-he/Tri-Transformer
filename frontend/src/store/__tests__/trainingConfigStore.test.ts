import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';

const server = setupServer(
  http.post('http://localhost:8002/api/v1/train/jobs/start', () =>
    HttpResponse.json({ jobId: 'test-job-1' })
  ),
  http.get('http://localhost:8002/api/v1/train/jobs/progress', () =>
    HttpResponse.json({
      jobId: 'test-job-1',
      phase: 0,
      step: 100,
      maxSteps: 1000,
      loss: 2.5,
      lr: 1e-4,
      status: 'running',
    })
  ),
  http.get('http://localhost:8002/api/v1/train/jobs/models', () =>
    HttpResponse.json({
      models: [
        { id: 'qwen2-audio', name: 'Qwen2-Audio-7B', type: 'input' },
        { id: 'llama3-8b', name: 'Llama-3-8B', type: 'output' },
      ],
    })
  )
);

beforeEach(() => {
  server.listen({ onUnhandledRequest: 'error' });
  vi.useFakeTimers();
});

afterEach(async () => {
  vi.useRealTimers();
  server.resetHandlers();
  server.close();
  const { useTrainingConfigStore } = await import('@/store/trainingConfigStore');
  useTrainingConfigStore.getState().reset();
});

describe('trainingConfigStore', () => {
  it('initial state has default config values', async () => {
    const { useTrainingConfigStore } = await import('@/store/trainingConfigStore');
    const state = useTrainingConfigStore.getState();
    expect(state.config.learning_rate).toBeGreaterThan(0);
    expect(state.config.batch_size).toBeGreaterThan(0);
    expect(state.currentJobId).toBeNull();
    expect(state.loading).toBe(false);
    expect(state.error).toBeNull();
  });

  it('setConfig updates specific config fields', async () => {
    const { useTrainingConfigStore } = await import('@/store/trainingConfigStore');
    useTrainingConfigStore.getState().setConfig({ learning_rate: 5e-5, phase: 1 });
    const config = useTrainingConfigStore.getState().config;
    expect(config.learning_rate).toBe(5e-5);
    expect(config.phase).toBe(1);
  });

  it('startTraining calls POST /training/start and sets currentJobId', async () => {
    const { useTrainingConfigStore } = await import('@/store/trainingConfigStore');
    await useTrainingConfigStore.getState().startTraining();
    expect(useTrainingConfigStore.getState().currentJobId).toBe('test-job-1');
    expect(useTrainingConfigStore.getState().loading).toBe(false);
  });

  it('fetchProgress updates progress state', async () => {
    const { useTrainingConfigStore } = await import('@/store/trainingConfigStore');
    await useTrainingConfigStore.getState().fetchProgress();
    const state = useTrainingConfigStore.getState();
    expect(state.progress).not.toBeNull();
    expect(state.progress?.step).toBe(100);
    expect(state.progress?.maxSteps).toBe(1000);
    expect(state.progress?.status).toBe('running');
  });

  it('fetchAvailableModels updates availableModels list', async () => {
    const { useTrainingConfigStore } = await import('@/store/trainingConfigStore');
    await useTrainingConfigStore.getState().fetchAvailableModels();
    const state = useTrainingConfigStore.getState();
    expect(state.availableModels).toHaveLength(2);
    expect(state.availableModels[0].type).toBe('input');
    expect(state.availableModels[1].type).toBe('output');
  });

  it('startTraining failure sets error state', async () => {
    server.use(
      http.post('http://localhost:8002/api/v1/train/jobs/start', () =>
        HttpResponse.json({ detail: 'Internal error' }, { status: 500 })
      )
    );
    const { useTrainingConfigStore } = await import('@/store/trainingConfigStore');
    await useTrainingConfigStore.getState().startTraining();
    const state = useTrainingConfigStore.getState();
    expect(state.error).not.toBeNull();
    expect(state.loading).toBe(false);
  });
});
