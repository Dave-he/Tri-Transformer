import apiClient from './client';
import type { Document, SearchResult } from '@/types/api';

export const getDocumentsApi = async (): Promise<{ documents: Document[] }> => {
  const { data } = await apiClient.get<{ documents: Document[] }>('/documents');
  return data;
};

export const uploadDocumentApi = async (file: File): Promise<Document> => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post<Document>('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const deleteDocumentApi = async (id: string): Promise<{ message: string }> => {
  const { data } = await apiClient.delete<{ message: string }>(`/documents/${id}`);
  return data;
};

export const searchDocumentsApi = async (query: string, topK = 10): Promise<{ results: SearchResult[] }> => {
  const { data } = await apiClient.post<{ results: SearchResult[] }>('/documents/search', { query, topK });
  return data;
};
