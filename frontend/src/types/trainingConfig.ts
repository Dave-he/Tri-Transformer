export interface TrainingConfig {
  readonly i_model_id: string;
  readonly o_model_id: string;
  readonly learning_rate: number;
  readonly batch_size: number;
  readonly max_steps: number;
  readonly phase: 0 | 1 | 2;
}

export interface TrainingJob {
  readonly jobId: string;
}

export interface TrainingProgress {
  readonly jobId: string;
  readonly phase: number;
  readonly step: number;
  readonly maxSteps: number;
  readonly loss: number;
  readonly lr: number;
  readonly status: 'running' | 'completed' | 'failed';
}

export interface AvailableModel {
  readonly id: string;
  readonly name: string;
  readonly type: 'input' | 'output';
}

export interface TrainingConfigState {
  config: Omit<TrainingConfig, never>;
  currentJobId: string | null;
  progress: TrainingProgress | null;
  availableModels: AvailableModel[];
  loading: boolean;
  error: string | null;
  setConfig: (partial: Partial<TrainingConfig>) => void;
  startTraining: () => Promise<void>;
  fetchProgress: () => Promise<void>;
  fetchAvailableModels: () => Promise<void>;
  reset: () => void;
}
