import { apiClient } from './client';
import type { TrainingConfig, TrainingJob, TrainingProgress, AvailableModel } from '@/types/trainingConfig';

export const startTrainingApi = async (config: TrainingConfig): Promise<TrainingJob> => {
  const { data } = await apiClient.post<TrainingJob>('/train/jobs/start', config);
  return data;
};

export const getTrainingProgressApi = async (): Promise<TrainingProgress> => {
  const { data } = await apiClient.get<TrainingProgress>('/train/jobs/progress');
  return data;
};

export const getAvailableModelsApi = async (): Promise<{ models: AvailableModel[] }> => {
  const { data } = await apiClient.get<{ models: AvailableModel[] }>('/train/jobs/models');
  return data;
};
