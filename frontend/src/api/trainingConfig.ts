import { apiClient } from './client';
import type { TrainingConfig, TrainingJob, TrainingProgress, AvailableModel } from '@/types/trainingConfig';

export const startTrainingApi = async (config: TrainingConfig): Promise<TrainingJob> => {
  const { data } = await apiClient.post<TrainingJob>('/training/start', config);
  return data;
};

export const getTrainingProgressApi = async (): Promise<TrainingProgress> => {
  const { data } = await apiClient.get<TrainingProgress>('/training/progress');
  return data;
};

export const getAvailableModelsApi = async (): Promise<{ models: AvailableModel[] }> => {
  const { data } = await apiClient.get<{ models: AvailableModel[] }>('/models/available');
  return data;
};
