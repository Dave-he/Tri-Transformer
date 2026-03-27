import React, { useState } from 'react';
import { Form, Input, Button, Alert } from 'antd';
import { useNavigate, Link } from 'react-router-dom';
import { AuthLayout } from '@/layouts/AuthLayout';
import { useAuth } from '@/hooks/useAuth';

const LoginPage: React.FC = () => {
  const { login, loading } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  const onFinish = async (values: { username: string; password: string }) => {
    setError(null);
    try {
      await login(values.username, values.password);
      navigate('/chat');
    } catch {
      setError('用户名或密码错误，请重试');
    }
  };

  return (
    <AuthLayout>
      {error && <Alert message={error} type="error" style={{ marginBottom: 16 }} />}
      <Form layout="vertical" onFinish={onFinish} autoComplete="off">
        <Form.Item name="username" label="用户名" rules={[{ required: true, message: '请输入用户名' }]}>
          <Input placeholder="请输入用户名" />
        </Form.Item>
        <Form.Item name="password" label="密码" rules={[{ required: true, message: '请输入密码' }]}>
          <Input.Password placeholder="请输入密码" />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>
            登录
          </Button>
        </Form.Item>
        <div style={{ textAlign: 'center' }}>
          还没有账号？<Link to="/register">立即注册</Link>
        </div>
      </Form>
    </AuthLayout>
  );
};

export default LoginPage;
