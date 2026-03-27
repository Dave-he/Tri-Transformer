import { setupServer } from 'msw/node';
import { authHandlers } from './handlers/auth';
import { conversationHandlers } from './handlers/conversations';
import { documentHandlers } from './handlers/documents';
import { trainingHandlers } from './handlers/training';

export const server = setupServer(
  ...authHandlers,
  ...conversationHandlers,
  ...documentHandlers,
  ...trainingHandlers
);
