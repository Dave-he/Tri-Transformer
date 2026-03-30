import { setupServer } from 'msw/node';
import { authHandlers } from './handlers/auth';
import { conversationHandlers } from './handlers/conversations';
import { documentHandlers } from './handlers/documents';
import { trainingHandlers } from './handlers/training';
import { webrtcHandlers } from './handlers/webrtc';
import { trainingConfigHandlers } from './handlers/trainingConfig';

export const server = setupServer(
  ...authHandlers,
  ...conversationHandlers,
  ...documentHandlers,
  ...trainingHandlers,
  ...webrtcHandlers,
  ...trainingConfigHandlers
);
