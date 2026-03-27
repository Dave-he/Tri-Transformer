import apiClient from './client';
import type { LoginResponse, User } from '@/types/api';

export const loginApi = async (username: string, password: string): Promise<LoginResponse> => {
  const { data } = await apiClient.post<LoginResponse>('/auth/login', { username, password });
  return data;
};

export const registerApi = async (username: string, password: string, email: string): Promise<{ user: User }> => {
  const { data } = await apiClient.post<{ user: User }>('/auth/register', { username, password, email });
  return data;
};

export const logoutApi = async (): Promise<void> => {
  await apiClient.post('/auth/logout');
};
