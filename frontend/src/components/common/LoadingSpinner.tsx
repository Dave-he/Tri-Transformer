import React from 'react';
import { Spin } from 'antd';

interface LoadingSpinnerProps {
  size?: 'small' | 'default' | 'large';
  tip?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ size = 'default', tip }) => {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 40 }}>
      <Spin size={size} tip={tip} />
    </div>
  );
};
