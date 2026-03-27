import type { User, Conversation, Message, Document, SearchResult, Metrics, TrainingStatus } from './api';

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, email: string) => Promise<void>;
  logout: () => void;
}

export interface ConversationState {
  conversations: Conversation[];
  currentConversationId: string | null;
  messages: Message[];
  loading: boolean;
  error: string | null;
  fetchConversations: () => Promise<void>;
  createConversation: (title?: string) => Promise<void>;
  setActiveConversation: (id: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
}

export interface DocumentState {
  documents: Document[];
  uploadProgress: number | null;
  searchResults: SearchResult[];
  loading: boolean;
  error: string | null;
  fetchDocuments: () => Promise<void>;
  upload: (file: File) => Promise<void>;
  deleteDocument: (id: string) => Promise<void>;
  search: (query: string, topK?: number) => Promise<void>;
}

export interface MetricsState {
  metrics: Metrics | null;
  trainingStatus: TrainingStatus | null;
  loading: boolean;
  fetchMetrics: () => Promise<void>;
  fetchStatus: () => Promise<void>;
}
