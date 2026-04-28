export interface User {
  readonly id: string;
  readonly username: string;
  readonly email: string;
}

export interface Conversation {
  readonly id: string;
  readonly title: string;
  readonly status: string;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly messageCount: number;
}

export interface PaginationInfo {
  readonly page: number;
  readonly pageSize: number;
  readonly total: number;
  readonly totalPages: number;
}

export interface MessageSource {
  readonly document: string;
  readonly chunk: string;
  readonly score: number;
}

export interface Message {
  readonly id: string;
  readonly role: 'user' | 'assistant';
  readonly content: string;
  readonly sources: MessageSource[];
  readonly createdAt: string;
  readonly hallucinationDetected?: boolean;
}

export type DocumentStatus = 'processing' | 'ready' | 'failed';

export interface Document {
  readonly id: string;
  readonly name: string;
  readonly type: string;
  readonly size: number;
  readonly status: DocumentStatus;
  readonly createdAt: string;
}

export interface SearchResult {
  readonly document: string;
  readonly chunk: string;
  readonly score: number;
}

export interface MetricsCurrent {
  readonly retrievalAccuracy: number;
  readonly bleuScore: number;
  readonly hallucinationRate: number;
}

export interface MetricsHistory {
  readonly timestamp: string;
  readonly retrievalAccuracy: number;
  readonly bleuScore: number;
  readonly hallucinationRate: number;
}

export interface Metrics {
  readonly current: MetricsCurrent;
  readonly history: MetricsHistory[];
}

export interface TrainingStatus {
  readonly phase: string;
  readonly progress: number;
  readonly eta: string;
  readonly message: string;
}

export interface ApiResponse<T> {
  readonly data: T;
  readonly message?: string;
}

export interface LoginResponse {
  readonly access_token: string;
  readonly token_type: string;
}

export interface ModelStatus {
  readonly status: string;
  readonly model_loaded: boolean;
  readonly mock_mode: boolean;
}

export interface ModelInfo {
  readonly model_type: string;
  readonly config: Record<string, unknown>;
  readonly device: string;
}

export interface DocumentStatusResponse {
  readonly document_id: string;
  readonly status: DocumentStatus;
  readonly progress: number;
  readonly chunk_count: number;
}

export interface TrainConfigPreset {
  readonly name: string;
  readonly description: string;
  readonly learning_rate: number;
  readonly batch_size: number;
  readonly epochs: number;
  readonly lora_rank: number;
  readonly lora_alpha: number;
}
