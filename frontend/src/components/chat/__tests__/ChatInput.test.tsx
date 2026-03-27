import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatInput } from '../ChatInput';

describe('ChatInput', () => {
  it('calls onSend when Enter key pressed', () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} loading={false} />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Test message' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSend).toHaveBeenCalledWith('Test message');
  });

  it('does not call onSend on Shift+Enter', () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} loading={false} />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Line 1' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

    expect(onSend).not.toHaveBeenCalled();
  });

  it('disables send button when loading is true', () => {
    render(<ChatInput onSend={vi.fn()} loading={true} />);

    const button = screen.getByRole('button');
    expect(button).toHaveProperty('disabled', true);
  });

  it('clears input after sending', () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} loading={false} />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Send this' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect((textarea as HTMLTextAreaElement).value).toBe('');
  });

  it('does not call onSend with empty input', () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} loading={false} />);

    const textarea = screen.getByRole('textbox');
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSend).not.toHaveBeenCalled();
  });
});
