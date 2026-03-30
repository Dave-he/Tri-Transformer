import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatModeTabs } from '@/components/chat/ChatModeTabs';

describe('ChatModeTabs', () => {
  it('renders default text mode with children', () => {
    const onModeChange = vi.fn();
    render(
      <ChatModeTabs mode="text" onModeChange={onModeChange}>
        <div>Text content</div>
      </ChatModeTabs>
    );
    expect(screen.getByText('Text content')).toBeInTheDocument();
  });

  it('shows all three tabs', () => {
    render(
      <ChatModeTabs mode="text" onModeChange={vi.fn()}>
        <div>content</div>
      </ChatModeTabs>
    );
    expect(screen.getByRole('tab', { name: /文本/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /语音/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /视频/i })).toBeInTheDocument();
  });

  it('calls onModeChange with correct mode when audio tab clicked', () => {
    const onModeChange = vi.fn();
    render(
      <ChatModeTabs mode="text" onModeChange={onModeChange}>
        <div>content</div>
      </ChatModeTabs>
    );
    fireEvent.click(screen.getByRole('tab', { name: /语音/i }));
    expect(onModeChange).toHaveBeenCalledWith('audio');
  });

  it('calls onModeChange with correct mode when video tab clicked', () => {
    const onModeChange = vi.fn();
    render(
      <ChatModeTabs mode="text" onModeChange={onModeChange}>
        <div>content</div>
      </ChatModeTabs>
    );
    fireEvent.click(screen.getByRole('tab', { name: /视频/i }));
    expect(onModeChange).toHaveBeenCalledWith('video');
  });
});
