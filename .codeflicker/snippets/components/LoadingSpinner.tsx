/**
 * 可复用组件 - LoadingSpinner
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies react, antd
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/common/LoadingSpinner.tsx
 * 评分: 5.60
 * 复杂度: 2
 */

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
