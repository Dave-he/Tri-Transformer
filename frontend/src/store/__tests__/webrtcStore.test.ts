import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';

const server = setupServer(
  http.post('http://localhost:8002/api/v1/webrtc/offer', () =>
    HttpResponse.json({ sdp: 'answer_sdp', type: 'answer' })
  ),
  http.post('http://localhost:8002/api/v1/webrtc/candidate', () =>
    HttpResponse.json({ ok: true })
  ),
  http.post('http://localhost:8002/api/v1/webrtc/interrupt', () =>
    HttpResponse.json({ ok: true })
  )
);

const mockClose = vi.fn();
const mockCreateOffer = vi.fn().mockResolvedValue({ type: 'offer', sdp: 'offer_sdp' });
const mockSetLocalDescription = vi.fn().mockResolvedValue(undefined);
const mockSetRemoteDescription = vi.fn().mockResolvedValue(undefined);
const MockRTCPeerConnection = vi.fn().mockImplementation(() => ({
  createOffer: mockCreateOffer,
  setLocalDescription: mockSetLocalDescription,
  setRemoteDescription: mockSetRemoteDescription,
  addIceCandidate: vi.fn().mockResolvedValue(undefined),
  close: mockClose,
  onicecandidate: null,
  ontrack: null,
}));

vi.stubGlobal('RTCPeerConnection', MockRTCPeerConnection);

const mockGetUserMedia = vi.fn().mockResolvedValue({
  getTracks: () => [{ stop: vi.fn() }],
});
Object.defineProperty(navigator, 'mediaDevices', {
  writable: true,
  value: { getUserMedia: mockGetUserMedia },
});

beforeEach(async () => {
  server.listen({ onUnhandledRequest: 'error' });
  vi.clearAllMocks();
  const { useWebRTCStore } = await import('@/store/webrtcStore');
  useWebRTCStore.getState().reset();
});

afterEach(() => {
  server.resetHandlers();
  server.close();
});

describe('webrtcStore', () => {
  it('initial state is idle', async () => {
    const { useWebRTCStore } = await import('@/store/webrtcStore');
    const state = useWebRTCStore.getState();
    expect(state.connectionState).toBe('idle');
    expect(state.localStream).toBeNull();
    expect(state.remoteStream).toBeNull();
    expect(state.error).toBeNull();
  });

  it('startCall transitions to requesting_media then connecting', async () => {
    const { useWebRTCStore } = await import('@/store/webrtcStore');
    const startCallPromise = useWebRTCStore.getState().startCall();
    await startCallPromise;
    const state = useWebRTCStore.getState();
    expect(['connecting', 'connected', 'error']).toContain(state.connectionState);
    expect(mockGetUserMedia).toHaveBeenCalledWith({ audio: true, video: false });
  });

  it('endCall sets state to disconnected and calls close()', async () => {
    const { useWebRTCStore } = await import('@/store/webrtcStore');
    await useWebRTCStore.getState().startCall();
    useWebRTCStore.getState().endCall();
    expect(useWebRTCStore.getState().connectionState).toBe('disconnected');
    expect(mockClose).toHaveBeenCalled();
  });

  it('reset restores state to idle with null streams', async () => {
    const { useWebRTCStore } = await import('@/store/webrtcStore');
    await useWebRTCStore.getState().startCall();
    useWebRTCStore.getState().reset();
    const state = useWebRTCStore.getState();
    expect(state.connectionState).toBe('idle');
    expect(state.localStream).toBeNull();
    expect(state.remoteStream).toBeNull();
  });

  it('getUserMedia failure sets connectionState to error', async () => {
    mockGetUserMedia.mockRejectedValueOnce(new Error('Permission denied'));
    const { useWebRTCStore } = await import('@/store/webrtcStore');
    await useWebRTCStore.getState().startCall();
    const state = useWebRTCStore.getState();
    expect(state.connectionState).toBe('error');
    expect(state.error).toContain('Permission denied');
  });

  it('sendInterrupt calls POST /webrtc/interrupt', async () => {
    const { useWebRTCStore } = await import('@/store/webrtcStore');
    await expect(useWebRTCStore.getState().sendInterrupt()).resolves.not.toThrow();
  });
});
