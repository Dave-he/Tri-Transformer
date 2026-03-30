import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TrainingStatusCard } from '../TrainingStatusCard';
import type { TrainingStatus } from '@/types/api';

vi.mock('antd', async () => {
  const actual = await vi.importActual<typeof import('antd')>('antd');
  return actual;
});

const mockStatus: TrainingStatus = {
  phase: 'Stage 2: C-Transformer Training',
  progress: 45,
  eta: '约 2 小时后完成',
  message: '正在训练控制中枢分支...',
};

describe('TrainingStatusCard', () => {
  it('renders fallback when status is null', () => {
    render(<TrainingStatusCard status={null} />);
    expect(screen.getByText('暂无训练状态')).toBeDefined();
  });

  it('renders training phase when status is provided', () => {
    render(<TrainingStatusCard status={mockStatus} />);
    expect(screen.getByText('Stage 2: C-Transformer Training')).toBeDefined();
  });

  it('renders eta and message', () => {
    render(<TrainingStatusCard status={mockStatus} />);
    expect(screen.getByText('约 2 小时后完成')).toBeDefined();
    expect(screen.getByText('正在训练控制中枢分支...')).toBeDefined();
  });

  it('shows progress percentage', () => {
    render(<TrainingStatusCard status={mockStatus} />);
    expect(screen.getByText(/45%/)).toBeDefined();
  });
});
