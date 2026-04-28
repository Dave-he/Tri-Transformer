import apiClient from './client';
import type { Metrics, TrainingStatus, TrainConfigPreset } from '@/types/api';

export const getMetricsApi = async (): Promise<Metrics> => {
  const { data } = await apiClient.get<Metrics>('/metrics');
  return data;
};

export const getTrainingStatusApi = async (): Promise<TrainingStatus> => {
  const { data } = await apiClient.get<TrainingStatus>('/training/status');
  return data;
};

export const getTrainConfigsApi = async (): Promise<{ configs: TrainConfigPreset[] }> => {
  const { data } = await apiClient.get<{ configs: TrainConfigPreset[] }>('/train/configs');
  return data;
};
