import { useAuthStore } from '@/store/authStore';

export const useAuth = () => {
  const { user, token, isAuthenticated, loading, error, login, register, logout } = useAuthStore();
  return { user, token, isAuthenticated, loading, error, login, register, logout };
};
