import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ModelPluginSelector } from '@/components/training/ModelPluginSelector';
import type { AvailableModel } from '@/types/trainingConfig';

const availableModels: AvailableModel[] = [
  { id: 'qwen2-audio', name: 'Qwen2-Audio-7B', type: 'input' },
  { id: 'qwen2-vl', name: 'Qwen2-VL-7B', type: 'input' },
  { id: 'llama3-8b', name: 'Llama-3-8B', type: 'output' },
];

describe('ModelPluginSelector', () => {
  it('renders I-Transformer and O-Transformer labels', () => {
    render(
      <ModelPluginSelector
        availableModels={availableModels}
        iModelId=""
        oModelId=""
        onIModelChange={vi.fn()}
        onOModelChange={vi.fn()}
      />
    );
    expect(screen.getByText(/I-Transformer/i)).toBeInTheDocument();
    expect(screen.getByText(/O-Transformer/i)).toBeInTheDocument();
  });

  it('calls onIModelChange when I-model input changes', () => {
    const onIModelChange = vi.fn();
    render(
      <ModelPluginSelector
        availableModels={availableModels}
        iModelId=""
        oModelId=""
        onIModelChange={onIModelChange}
        onOModelChange={vi.fn()}
      />
    );
    const inputs = screen.getAllByRole('textbox');
    fireEvent.change(inputs[0], { target: { value: 'custom/model-id' } });
    expect(onIModelChange).toHaveBeenCalledWith('custom/model-id');
  });

  it('calls onOModelChange when O-model input changes', () => {
    const onOModelChange = vi.fn();
    render(
      <ModelPluginSelector
        availableModels={availableModels}
        iModelId=""
        oModelId=""
        onIModelChange={vi.fn()}
        onOModelChange={onOModelChange}
      />
    );
    const inputs = screen.getAllByRole('textbox');
    fireEvent.change(inputs[1], { target: { value: 'custom/output-model' } });
    expect(onOModelChange).toHaveBeenCalledWith('custom/output-model');
  });

  it('displays current iModelId and oModelId values', () => {
    render(
      <ModelPluginSelector
        availableModels={availableModels}
        iModelId="Qwen/Qwen2-Audio-7B"
        oModelId="meta-llama/Llama-3-8B"
        onIModelChange={vi.fn()}
        onOModelChange={vi.fn()}
      />
    );
    expect(screen.getByDisplayValue('Qwen/Qwen2-Audio-7B')).toBeInTheDocument();
    expect(screen.getByDisplayValue('meta-llama/Llama-3-8B')).toBeInTheDocument();
  });
});
