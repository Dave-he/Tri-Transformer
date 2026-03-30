import { create } from 'zustand';
import { startTrainingApi, getTrainingProgressApi, getAvailableModelsApi } from '@/api/trainingConfig';
import type { TrainingConfig, TrainingProgress, AvailableModel, TrainingConfigState } from '@/types/trainingConfig';

const defaultConfig: TrainingConfig = {
  i_model_id: '',
  o_model_id: '',
  learning_rate: 1e-4,
  batch_size: 8,
  max_steps: 1000,
  phase: 0,
};

export const useTrainingConfigStore = create<TrainingConfigState>((set, get) => ({
  config: { ...defaultConfig },
  currentJobId: null,
  progress: null as TrainingProgress | null,
  availableModels: [] as AvailableModel[],
  loading: false,
  error: null as string | null,

  setConfig: (partial: Partial<TrainingConfig>) => {
    set((state) => ({ config: { ...state.config, ...partial } }));
  },

  startTraining: async () => {
    set({ loading: true, error: null });
    try {
      const job = await startTrainingApi(get().config);
      set({ currentJobId: job.jobId, loading: false });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Training failed';
      set({ error: message, loading: false });
    }
  },

  fetchProgress: async () => {
    try {
      const progress = await getTrainingProgressApi();
      set({ progress });
    } catch {
    }
  },

  fetchAvailableModels: async () => {
    try {
      const result = await getAvailableModelsApi();
      set({ availableModels: result.models });
    } catch {
    }
  },

  reset: () => {
    set({
      config: { ...defaultConfig },
      currentJobId: null,
      progress: null,
      availableModels: [],
      loading: false,
      error: null,
    });
  },
}));
