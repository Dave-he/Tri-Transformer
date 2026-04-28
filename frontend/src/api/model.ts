import apiClient from './client';
import type { ModelStatus, ModelInfo } from '@/types/api';

export const getModelStatusApi = async (): Promise<ModelStatus> => {
  const { data } = await apiClient.get<ModelStatus>('/model/status');
  return data;
};

export const loadModelApi = async (modelPath?: string): Promise<{ message: string }> => {
  const { data } = await apiClient.post<{ message: string }>('/model/load', { model_path: modelPath });
  return data;
};

export const getModelInfoApi = async (): Promise<ModelInfo> => {
  const { data } = await apiClient.get<ModelInfo>('/model/info');
  return data;
};
