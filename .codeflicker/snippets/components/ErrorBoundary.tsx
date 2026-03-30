/**
 * 可复用组件 - ErrorBoundary
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies react, antd
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/common/ErrorBoundary.tsx
 * 评分: 4.45
 * 复杂度: 4
 */

import React from 'react';
import { Result, Button } from 'antd';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <Result
          status="error"
          title="页面发生错误"
          subTitle={this.state.error?.message ?? '未知错误'}
          extra={
            <Button type="primary" onClick={() => this.setState({ hasError: false })}>
              重试
            </Button>
          }
        />
      );
    }
    return this.props.children;
  }
}
