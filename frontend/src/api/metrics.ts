import apiClient from './client';
import type { Metrics, TrainingStatus } from '@/types/api';

export const getMetricsApi = async (): Promise<Metrics> => {
  const { data } = await apiClient.get<Metrics>('/metrics');
  return data;
};

export const getTrainingStatusApi = async (): Promise<TrainingStatus> => {
  const { data } = await apiClient.get<TrainingStatus>('/training/status');
  return data;
};
