import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';

vi.mock('../../store/authStore', () => ({
  useAuthStore: vi.fn(),
}));

import { AuthGuard } from '../MainLayout';
import { useAuthStore } from '../../store/authStore';

const notAuthState = {
  isAuthenticated: false,
  user: null,
  token: null,
  loading: false,
  error: null,
  login: vi.fn(),
  register: vi.fn(),
  logout: vi.fn(),
};

const authState = {
  ...notAuthState,
  isAuthenticated: true,
  user: { id: '1', username: 'test', email: 'test@test.com' },
  token: 'token',
};

describe('AuthGuard', () => {
  beforeEach(() => {
    cleanup();
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('does not render protected content when not authenticated', () => {
    vi.mocked(useAuthStore as unknown as (s?: unknown) => unknown).mockImplementation(
      (selector: unknown) => typeof selector === 'function' ? (selector as (s: unknown) => unknown)(notAuthState) : notAuthState
    );

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Routes>
          <Route path="/login" element={<div>Login Page</div>} />
          <Route
            path="/chat"
            element={
              <AuthGuard>
                <div>Protected Content</div>
              </AuthGuard>
            }
          />
        </Routes>
      </MemoryRouter>
    );

    expect(screen.queryByText('Protected Content')).toBeNull();
    expect(screen.getByText('Login Page')).toBeDefined();
  });

  it('renders children when authenticated', () => {
    vi.mocked(useAuthStore as unknown as (s?: unknown) => unknown).mockImplementation(
      (selector: unknown) => typeof selector === 'function' ? (selector as (s: unknown) => unknown)(authState) : authState
    );

    render(
      <MemoryRouter>
        <AuthGuard>
          <div>Protected Content</div>
        </AuthGuard>
      </MemoryRouter>
    );

    expect(screen.getByText('Protected Content')).toBeDefined();
  });
});
