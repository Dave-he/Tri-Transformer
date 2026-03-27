import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../../api/auth', () => ({
  loginApi: vi.fn(),
  registerApi: vi.fn(),
  logoutApi: vi.fn().mockResolvedValue({ message: 'ok' }),
}));

describe('authStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('initial state: isAuthenticated is false', async () => {
    const { useAuthStore } = await import('../authStore');
    const { getState } = useAuthStore;
    expect(getState().isAuthenticated).toBe(false);
    expect(getState().user).toBeNull();
    expect(getState().token).toBeNull();
  });

  it('login success sets isAuthenticated to true and stores token', async () => {
    const { loginApi } = await import('../../api/auth');
    const { useAuthStore } = await import('../authStore');

    vi.mocked(loginApi).mockResolvedValue({
      token: 'test-token-123',
      user: { id: '1', username: 'testuser', email: 'test@test.com' },
    });

    await useAuthStore.getState().login('testuser', 'password');

    expect(useAuthStore.getState().isAuthenticated).toBe(true);
    expect(useAuthStore.getState().token).toBe('test-token-123');
    expect(useAuthStore.getState().user?.username).toBe('testuser');
  });

  it('login failure sets error state', async () => {
    const { loginApi } = await import('../../api/auth');
    const { useAuthStore } = await import('../authStore');

    vi.mocked(loginApi).mockRejectedValue(new Error('Invalid credentials'));

    await expect(useAuthStore.getState().login('wrong', 'wrong')).rejects.toThrow();
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });

  it('logout clears token and user state', async () => {
    const { loginApi } = await import('../../api/auth');
    const { useAuthStore } = await import('../authStore');

    vi.mocked(loginApi).mockResolvedValue({
      token: 'test-token',
      user: { id: '1', username: 'test', email: 'test@test.com' },
    });

    await useAuthStore.getState().login('test', 'pass');
    useAuthStore.getState().logout();

    expect(useAuthStore.getState().isAuthenticated).toBe(false);
    expect(useAuthStore.getState().token).toBeNull();
    expect(useAuthStore.getState().user).toBeNull();
  });
});
