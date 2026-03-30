import { setupWorker } from 'msw/browser';
import { authHandlers } from './handlers/auth';
import { conversationHandlers } from './handlers/conversations';
import { documentHandlers } from './handlers/documents';
import { trainingHandlers } from './handlers/training';
import { trainingConfigHandlers } from './handlers/trainingConfig';
import { webrtcHandlers } from './handlers/webrtc';

export const worker = setupWorker(
  ...authHandlers,
  ...conversationHandlers,
  ...documentHandlers,
  ...trainingHandlers,
  ...trainingConfigHandlers,
  ...webrtcHandlers
);
