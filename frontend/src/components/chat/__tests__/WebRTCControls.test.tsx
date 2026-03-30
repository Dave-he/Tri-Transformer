import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { WebRTCControls } from '@/components/chat/WebRTCControls';

describe('WebRTCControls', () => {
  it('shows start call button when idle', () => {
    const onStartCall = vi.fn();
    render(
      <WebRTCControls
        connectionState="idle"
        onStartCall={onStartCall}
        onEndCall={vi.fn()}
        onInterrupt={vi.fn()}
        onStyleChange={vi.fn()}
      />
    );
    expect(screen.getByRole('button', { name: /开始通话/i })).toBeInTheDocument();
  });

  it('calls onStartCall when start button is clicked', () => {
    const onStartCall = vi.fn();
    render(
      <WebRTCControls
        connectionState="idle"
        onStartCall={onStartCall}
        onEndCall={vi.fn()}
        onInterrupt={vi.fn()}
        onStyleChange={vi.fn()}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /开始通话/i }));
    expect(onStartCall).toHaveBeenCalledTimes(1);
  });

  it('shows end call and interrupt buttons when connected', () => {
    render(
      <WebRTCControls
        connectionState="connected"
        onStartCall={vi.fn()}
        onEndCall={vi.fn()}
        onInterrupt={vi.fn()}
        onStyleChange={vi.fn()}
      />
    );
    expect(screen.getByRole('button', { name: /结束通话/i })).toBeInTheDocument();
    expect(screen.getAllByRole('button').length).toBeGreaterThanOrEqual(2);
    expect(document.querySelector('button span')?.textContent).toBeDefined();
  });

  it('calls onInterrupt when interrupt button is clicked', () => {
    const onInterrupt = vi.fn();
    render(
      <WebRTCControls
        connectionState="connected"
        onStartCall={vi.fn()}
        onEndCall={vi.fn()}
        onInterrupt={onInterrupt}
        onStyleChange={vi.fn()}
      />
    );
    const buttons = screen.getAllByRole('button');
    const interruptBtn = buttons.find(btn => btn.textContent?.includes('断'));
    expect(interruptBtn).toBeDefined();
    if (interruptBtn) fireEvent.click(interruptBtn);
    expect(onInterrupt).toHaveBeenCalledTimes(1);
  });

  it('shows connecting state when connecting', () => {
    render(
      <WebRTCControls
        connectionState="connecting"
        onStartCall={vi.fn()}
        onEndCall={vi.fn()}
        onInterrupt={vi.fn()}
        onStyleChange={vi.fn()}
      />
    );
    expect(screen.getAllByText(/连接中/i).length).toBeGreaterThanOrEqual(1);
  });
});
