export type WebRTCConnectionState =
  | 'idle'
  | 'requesting_media'
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'error';

export interface RTCSignalOffer {
  readonly sdp: string;
  readonly type: 'offer';
}

export interface RTCSignalAnswer {
  readonly sdp: string;
  readonly type: 'answer';
}

export interface RTCSignalCandidate {
  readonly candidate: string;
  readonly sdpMid?: string | null;
  readonly sdpMLineIndex?: number | null;
}

export interface WebRTCState {
  connectionState: WebRTCConnectionState;
  localStream: MediaStream | null;
  remoteStream: MediaStream | null;
  error: string | null;
  startCall: (video?: boolean) => Promise<void>;
  endCall: () => void;
  sendInterrupt: () => Promise<void>;
  reset: () => void;
}
