import { useDocumentStore } from '@/store/documentStore';

export const useDocuments = () => {
  return useDocumentStore();
};
