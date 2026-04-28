import { create } from 'zustand';
import {
  getDocumentsApi,
  uploadDocumentApi,
  deleteDocumentApi,
  searchDocumentsApi,
} from '@/api/documents';
import type { DocumentState } from '@/types/store';
import type { DocumentStatus } from '@/types/api';

export const useDocumentStore = create<DocumentState>((set) => ({
  documents: [],
  uploadProgress: null,
  searchResults: [],
  loading: false,
  error: null,

  fetchDocuments: async () => {
    set({ loading: true, error: null });
    try {
      const { documents } = await getDocumentsApi();
      set({ documents, loading: false });
    } catch {
      set({ loading: false, error: '加载文档列表失败' });
    }
  },

  upload: async (file: File) => {
    set({ uploadProgress: 0, error: null });
    try {
      set({ uploadProgress: 30 });
      const uploadResult = await uploadDocumentApi(file);
      set({ uploadProgress: 100 });
      await new Promise((r) => setTimeout(r, 200));
      set((state) => ({
        documents: [
          {
            id: uploadResult.document_id,
            name: file.name,
            type: file.name.split('.').pop() ?? '',
            size: file.size,
            status: uploadResult.status as DocumentStatus,
            createdAt: new Date().toISOString(),
          },
          ...state.documents,
        ],
        uploadProgress: null,
      }));
    } catch (err) {
      set({ uploadProgress: null, error: '上传失败' });
      throw err;
    }
  },

  deleteDocument: async (id: string) => {
    set({ loading: true, error: null });
    try {
      await deleteDocumentApi(id);
      set((state) => ({
        documents: state.documents.filter((d) => d.id !== id),
        loading: false,
      }));
    } catch {
      set({ loading: false, error: '删除失败' });
      throw new Error('删除失败');
    }
  },

  search: async (query: string, topK = 10) => {
    set({ loading: true, error: null });
    try {
      const { results } = await searchDocumentsApi(query, topK);
      set({ searchResults: results, loading: false });
    } catch {
      set({ loading: false, error: '检索失败' });
    }
  },
}));
