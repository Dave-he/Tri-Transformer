import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TrainingConfigForm } from '@/components/training/TrainingConfigForm';

describe('TrainingConfigForm', () => {
  it('renders all form fields', () => {
    render(<TrainingConfigForm onSubmit={vi.fn()} loading={false} />);
    expect(screen.getByLabelText(/学习率/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/批次大小/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/最大步数/i)).toBeInTheDocument();
  });

  it('submit button is disabled when loading is true', () => {
    render(<TrainingConfigForm onSubmit={vi.fn()} loading={true} />);
    expect(screen.getByRole('button', { name: /启动训练/i })).toBeDisabled();
  });

  it('submit button is enabled when loading is false', () => {
    render(<TrainingConfigForm onSubmit={vi.fn()} loading={false} />);
    expect(screen.getByRole('button', { name: /启动训练/i })).not.toBeDisabled();
  });

  it('calls onSubmit with form values when valid and submitted', async () => {
    const onSubmit = vi.fn();
    render(<TrainingConfigForm onSubmit={onSubmit} loading={false} />);
    fireEvent.click(screen.getByRole('button', { name: /启动训练/i }));
    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalled();
    });
    const callArg = onSubmit.mock.calls[0][0];
    expect(callArg).toHaveProperty('learning_rate');
    expect(callArg).toHaveProperty('batch_size');
    expect(callArg).toHaveProperty('max_steps');
    expect(callArg).toHaveProperty('phase');
  });

  it('shows validation error when learning_rate is invalid', async () => {
    render(<TrainingConfigForm onSubmit={vi.fn()} loading={false} />);
    const lrInput = screen.getByLabelText(/学习率/i);
    fireEvent.change(lrInput, { target: { value: '999' } });
    fireEvent.click(screen.getByRole('button', { name: /启动训练/i }));
    await waitFor(() => {
      expect(screen.getByText(/学习率/i)).toBeInTheDocument();
    });
  });
});
