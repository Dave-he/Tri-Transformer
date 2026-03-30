/**
 * 代码片段 - client
 * 
 * @category other
 * @tags other, react, typescript, dependencies
 * @dependencies axios
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/api/client.ts
 * 评分: 3.33
 * 复杂度: 5
 */

import axios from 'axios';

export const getAuthHeader = (): string | null => {
  try {
    const stored = localStorage.getItem('tri-transformer-auth');
    if (!stored) return null;
    const parsed = JSON.parse(stored) as { state?: { token?: string } };
    return parsed?.state?.token ?? null;
  } catch {
    return null;
  }
};

export const apiClient = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
});

apiClient.interceptors.request.use((config) => {
  const token = getAuthHeader();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      localStorage.removeItem('tri-transformer-auth');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;
