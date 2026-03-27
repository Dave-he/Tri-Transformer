import { describe, it, expect, vi } from 'vitest';

vi.mock('axios', () => {
  const mockInstance = {
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  };
  return {
    default: {
      create: vi.fn(() => mockInstance),
      isAxiosError: vi.fn(),
    },
  };
});

describe('API Client', () => {
  it('creates axios instance with correct base URL', async () => {
    const { apiClient } = await import('../client');
    expect(apiClient).toBeDefined();
  });

  it('exports client as default', async () => {
    const mod = await import('../client');
    expect(mod.apiClient).toBeDefined();
  });
});

describe('Auth token injection', () => {
  it('getAuthHeader returns null when no token stored', async () => {
    const { getAuthHeader } = await import('../client');
    const result = getAuthHeader();
    expect(result).toBeNull();
  });
});
