import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../../api/metrics', () => ({
  getMetricsApi: vi.fn(),
  getTrainingStatusApi: vi.fn(),
}));

describe('metricsStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
  });

  it('initial state: null metrics and trainingStatus', async () => {
    const { useMetricsStore } = await import('../metricsStore');
    const state = useMetricsStore.getState();
    expect(state.metrics).toBeNull();
    expect(state.trainingStatus).toBeNull();
  });

  it('fetchMetrics updates metrics state', async () => {
    const { getMetricsApi } = await import('../../api/metrics');
    const { useMetricsStore } = await import('../metricsStore');

    vi.mocked(getMetricsApi).mockResolvedValue({
      current: { retrievalAccuracy: 0.92, bleuScore: 0.78, hallucinationRate: 0.03 },
      history: [],
    });

    await useMetricsStore.getState().fetchMetrics();

    const metrics = useMetricsStore.getState().metrics;
    expect(metrics).not.toBeNull();
    expect(metrics?.current.retrievalAccuracy).toBe(0.92);
  });

  it('fetchStatus updates trainingStatus state', async () => {
    const { getTrainingStatusApi } = await import('../../api/metrics');
    const { useMetricsStore } = await import('../metricsStore');

    vi.mocked(getTrainingStatusApi).mockResolvedValue({
      phase: 'Stage 2',
      progress: 45,
      eta: '2h remaining',
      message: 'Training C-Transformer',
    });

    await useMetricsStore.getState().fetchStatus();

    const status = useMetricsStore.getState().trainingStatus;
    expect(status?.phase).toBe('Stage 2');
    expect(status?.progress).toBe(45);
  });
});
