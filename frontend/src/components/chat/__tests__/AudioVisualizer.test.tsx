import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';

const mockAnalyserDisconnect = vi.fn();
const mockConnect = vi.fn();
const mockAnalyserNode = {
  connect: mockConnect,
  disconnect: mockAnalyserDisconnect,
  fftSize: 2048,
  frequencyBinCount: 1024,
  getByteTimeDomainData: vi.fn((arr: Uint8Array) => arr.fill(128)),
};

const mockMediaStreamSourceConnect = vi.fn();
const mockMediaStreamSource = {
  connect: mockMediaStreamSourceConnect,
};

const mockAudioContextClose = vi.fn().mockResolvedValue(undefined);
const mockCreateAnalyser = vi.fn().mockReturnValue(mockAnalyserNode);
const mockCreateMediaStreamSource = vi.fn().mockReturnValue(mockMediaStreamSource);

const MockAudioContext = vi.fn().mockImplementation(() => ({
  createAnalyser: mockCreateAnalyser,
  createMediaStreamSource: mockCreateMediaStreamSource,
  close: mockAudioContextClose,
}));

vi.stubGlobal('AudioContext', MockAudioContext);

const mockGetContext = vi.fn().mockReturnValue({
  fillStyle: '',
  fillRect: vi.fn(),
  lineWidth: 1,
  strokeStyle: '',
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
});

Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  value: mockGetContext,
  writable: true,
});

const createMockStream = () =>
  ({
    getTracks: () => [],
    getAudioTracks: () => [],
    getVideoTracks: () => [],
    id: 'mock-stream-id',
    active: true,
    onaddtrack: null,
    onremovetrack: null,
    addTrack: vi.fn(),
    removeTrack: vi.fn(),
    getTrackById: vi.fn(),
    clone: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }) as unknown as MediaStream;

beforeEach(() => {
  vi.clearAllMocks();
  mockCreateAnalyser.mockReturnValue(mockAnalyserNode);
  mockCreateMediaStreamSource.mockReturnValue(mockMediaStreamSource);
  mockAudioContextClose.mockResolvedValue(undefined);
});

describe('AudioVisualizer', () => {
  it('renders canvas element when stream is provided', async () => {
    const { AudioVisualizer } = await import('@/components/chat/AudioVisualizer');
    const stream = createMockStream();
    render(<AudioVisualizer stream={stream} />);
    expect(document.querySelector('canvas')).toBeInTheDocument();
  });

  it('shows placeholder when stream is null', async () => {
    const { AudioVisualizer } = await import('@/components/chat/AudioVisualizer');
    render(<AudioVisualizer stream={null} />);
    expect(screen.getByText(/等待音频流/i)).toBeInTheDocument();
    expect(document.querySelector('canvas')).not.toBeInTheDocument();
  });

  it('creates AudioContext when stream is provided', async () => {
    const { AudioVisualizer } = await import('@/components/chat/AudioVisualizer');
    const stream = createMockStream();
    render(<AudioVisualizer stream={stream} />);
    expect(MockAudioContext).toHaveBeenCalled();
    expect(mockMediaStreamSourceConnect).toHaveBeenCalledWith(mockAnalyserNode);
  });

  it('calls AudioContext.close on unmount', async () => {
    const { AudioVisualizer } = await import('@/components/chat/AudioVisualizer');
    const stream = createMockStream();
    const { unmount } = render(<AudioVisualizer stream={stream} />);
    unmount();
    expect(mockAudioContextClose).toHaveBeenCalled();
  });
});
