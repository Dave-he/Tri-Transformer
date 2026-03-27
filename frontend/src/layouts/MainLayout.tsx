import React, { Suspense } from 'react';
import { Layout, Menu, Button, Spin } from 'antd';
import {
  MessageOutlined,
  FolderOutlined,
  BarChartOutlined,
  SettingOutlined,
  LogoutOutlined,
} from '@ant-design/icons';
import { Navigate, Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '@/store/authStore';

const { Sider, Content, Header } = Layout;

interface AuthGuardProps {
  children: React.ReactNode;
}

export const AuthGuard: React.FC<AuthGuardProps> = ({ children }) => {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
};

const menuItems = [
  { key: '/chat', icon: <MessageOutlined />, label: '对话助手' },
  { key: '/documents', icon: <FolderOutlined />, label: '知识库管理' },
  { key: '/training', icon: <SettingOutlined />, label: '训练监控' },
  { key: '/metrics', icon: <BarChartOutlined />, label: '性能指标' },
];

const MainLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { logout, user } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <AuthGuard>
      <Layout style={{ minHeight: '100vh' }}>
        <Sider theme="dark" width={220}>
          <div style={{ padding: '16px', color: '#fff', fontWeight: 'bold', fontSize: 16 }}>
            Tri-Transformer
          </div>
          <Menu
            theme="dark"
            mode="inline"
            selectedKeys={[location.pathname]}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
          />
        </Sider>
        <Layout>
          <Header
            style={{
              background: '#fff',
              padding: '0 24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              borderBottom: '1px solid #f0f0f0',
            }}
          >
            <span style={{ marginRight: 16, color: '#666' }}>{user?.username}</span>
            <Button icon={<LogoutOutlined />} onClick={handleLogout} type="text">
              退出
            </Button>
          </Header>
          <Content style={{ padding: 24, background: '#f5f5f5' }}>
            <Suspense fallback={<Spin size="large" style={{ display: 'block', marginTop: 100, textAlign: 'center' }} />}>
              <Outlet />
            </Suspense>
          </Content>
        </Layout>
      </Layout>
    </AuthGuard>
  );
};

export default MainLayout;
