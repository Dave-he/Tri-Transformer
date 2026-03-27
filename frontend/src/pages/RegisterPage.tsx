import React, { useState } from 'react';
import { Form, Input, Button, Alert } from 'antd';
import { useNavigate, Link } from 'react-router-dom';
import { AuthLayout } from '@/layouts/AuthLayout';
import { useAuth } from '@/hooks/useAuth';

const RegisterPage: React.FC = () => {
  const { register, loading } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  const onFinish = async (values: { username: string; email: string; password: string }) => {
    setError(null);
    try {
      await register(values.username, values.password, values.email);
      navigate('/login');
    } catch {
      setError('注册失败，请重试');
    }
  };

  return (
    <AuthLayout>
      {error && <Alert message={error} type="error" style={{ marginBottom: 16 }} />}
      <Form layout="vertical" onFinish={onFinish} autoComplete="off">
        <Form.Item name="username" label="用户名" rules={[{ required: true, message: '请输入用户名' }]}>
          <Input placeholder="请输入用户名" />
        </Form.Item>
        <Form.Item name="email" label="邮箱" rules={[{ required: true, type: 'email', message: '请输入有效邮箱' }]}>
          <Input placeholder="请输入邮箱" />
        </Form.Item>
        <Form.Item name="password" label="密码" rules={[{ required: true, min: 6, message: '密码至少 6 位' }]}>
          <Input.Password placeholder="请输入密码（至少 6 位）" />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>
            注册
          </Button>
        </Form.Item>
        <div style={{ textAlign: 'center' }}>
          已有账号？<Link to="/login">立即登录</Link>
        </div>
      </Form>
    </AuthLayout>
  );
};

export default RegisterPage;
