import { create } from 'zustand';
import { getMetricsApi, getTrainingStatusApi } from '@/api/training';
import type { MetricsState } from '@/types/store';

export const useMetricsStore = create<MetricsState>((set) => ({
  metrics: null,
  trainingStatus: null,
  loading: false,

  fetchMetrics: async () => {
    set({ loading: true });
    try {
      const metrics = await getMetricsApi();
      set({ metrics, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  fetchStatus: async () => {
    set({ loading: true });
    try {
      const trainingStatus = await getTrainingStatusApi();
      set({ trainingStatus, loading: false });
    } catch {
      set({ loading: false });
    }
  },
}));
