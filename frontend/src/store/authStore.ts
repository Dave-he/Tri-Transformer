import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { loginApi, registerApi, logoutApi } from '@/api/auth';
import type { AuthState } from '@/types/store';

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      loading: false,
      error: null,

      login: async (username: string, password: string) => {
        set({ loading: true, error: null });
        try {
          const { token } = await loginApi(username, password);
          set({ token, user: { id: '1', username, email: '' }, isAuthenticated: true, loading: false });
        } catch (err) {
          set({ loading: false, error: '用户名或密码错误', isAuthenticated: false });
          throw err;
        }
      },

      register: async (username: string, password: string, email: string) => {
        set({ loading: true, error: null });
        try {
          await registerApi(username, password, email);
          set({ loading: false });
        } catch (err) {
          set({ loading: false, error: '注册失败，请重试' });
          throw err;
        }
      },

      logout: () => {
        logoutApi().catch(() => undefined);
        set({ user: null, token: null, isAuthenticated: false, error: null });
      },
    }),
    {
      name: 'tri-transformer-auth',
      partialize: (state) => ({ token: state.token, user: state.user, isAuthenticated: state.isAuthenticated }),
    }
  )
);
