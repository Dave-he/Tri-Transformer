import { create } from 'zustand';
import { sendOffer, sendCandidate, sendInterrupt } from '@/api/webrtc';
import type { WebRTCConnectionState, WebRTCState } from '@/types/webrtc';

const initialState = {
  connectionState: 'idle' as WebRTCConnectionState,
  localStream: null as MediaStream | null,
  remoteStream: null as MediaStream | null,
  error: null as string | null,
};

let peerConnection: RTCPeerConnection | null = null;

export const useWebRTCStore = create<WebRTCState>((set, get) => ({
  ...initialState,

  startCall: async (video = false) => {
    set({ connectionState: 'requesting_media', error: null });

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video });
      set({ localStream: stream, connectionState: 'connecting' });

      peerConnection = new RTCPeerConnection();

      stream.getTracks().forEach((track) => {
        peerConnection?.addTrack?.(track, stream);
      });

      peerConnection.onicecandidate = async (event) => {
        if (event.candidate) {
          try {
            await sendCandidate({
              candidate: event.candidate.candidate,
              sdpMid: event.candidate.sdpMid,
              sdpMLineIndex: event.candidate.sdpMLineIndex,
            });
          } catch {
          }
        }
      };

      peerConnection.ontrack = (event) => {
        const [remoteStream] = event.streams;
        set({ remoteStream: remoteStream ?? null, connectionState: 'connected' });
      };

      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      const answer = await sendOffer({ sdp: offer.sdp ?? '', type: 'offer' });
      await peerConnection.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      set({ connectionState: 'error', error: message });
    }
  },

  endCall: () => {
    if (peerConnection) {
      peerConnection.close();
      peerConnection = null;
    }
    const { localStream } = get();
    localStream?.getTracks().forEach((track) => track.stop());
    set({ connectionState: 'disconnected', localStream: null });
  },

  sendInterrupt: async () => {
    await sendInterrupt();
  },

  reset: () => {
    if (peerConnection) {
      peerConnection.close();
      peerConnection = null;
    }
    const { localStream } = get();
    localStream?.getTracks().forEach((track) => track.stop());
    set(initialState);
  },
}));
