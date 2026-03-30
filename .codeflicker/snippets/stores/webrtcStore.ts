/**
 * 代码片段 - webrtcStore
 * 
 * @category stores
 * @tags stores, react, typescript, dependencies
 * @dependencies zustand
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/store/webrtcStore.ts
 * 评分: 4.71
 * 复杂度: 6
 */

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

// ... 更多实现