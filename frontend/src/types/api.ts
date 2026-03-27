export interface User {
  readonly id: string;
  readonly username: string;
  readonly email: string;
}

export interface Conversation {
  readonly id: string;
  readonly title: string;
  readonly createdAt: string;
  readonly updatedAt: string;
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
  readonly token: string;
  readonly user: User;
}
