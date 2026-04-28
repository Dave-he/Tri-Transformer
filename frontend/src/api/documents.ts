import apiClient from './client';
import type { Document, SearchResult, DocumentStatusResponse } from '@/types/api';

export const getDocumentsApi = async (): Promise<{ documents: Document[] }> => {
  const { data } = await apiClient.get<{ documents: Document[] }>('/knowledge/documents');
  return data;
};

export const uploadDocumentApi = async (file: File): Promise<{ document_id: string; status: string }> => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post<{ document_id: string; status: string }>('/knowledge/documents', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export const deleteDocumentApi = async (id: string): Promise<{ message: string }> => {
  const { data } = await apiClient.delete<{ message: string }>(`/knowledge/documents/${id}`);
  return data;
};

export const getDocumentStatusApi = async (docId: string): Promise<DocumentStatusResponse> => {
  const { data } = await apiClient.get<DocumentStatusResponse>(`/knowledge/documents/${docId}/status`);
  return data;
};

export const searchDocumentsApi = async (query: string, topK = 10): Promise<{ results: SearchResult[] }> => {
  const { data } = await apiClient.post<{ results: SearchResult[] }>('/knowledge/search', { query, top_k: topK });
  return data;
};
