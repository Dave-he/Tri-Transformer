import { apiClient } from './client';
import type { RTCSignalOffer, RTCSignalAnswer, RTCSignalCandidate } from '@/types/webrtc';

export const sendOffer = async (offer: RTCSignalOffer): Promise<RTCSignalAnswer> => {
  const { data } = await apiClient.post<RTCSignalAnswer>('/webrtc/offer', offer);
  return data;
};

export const sendCandidate = async (candidate: RTCSignalCandidate): Promise<{ ok: boolean }> => {
  const { data } = await apiClient.post<{ ok: boolean }>('/webrtc/candidate', { candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex });
  return data;
};

export const sendInterrupt = async (): Promise<{ ok: boolean }> => {
  const { data } = await apiClient.post<{ ok: boolean }>('/webrtc/interrupt');
  return data;
};
