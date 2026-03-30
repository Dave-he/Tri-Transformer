/**
 * 可复用组件 - EmptyState
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies react, antd
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/common/EmptyState.tsx
 * 评分: 5.91
 * 复杂度: 6
 */

import React from 'react';
import { Empty, Button } from 'antd';

interface EmptyStateProps {
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  image?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ description, actionLabel, onAction, image }) => {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 60 }}>
      <Empty
        image={image ?? Empty.PRESENTED_IMAGE_SIMPLE}
        description={description ?? '暂无数据'}
      >
        {actionLabel && onAction && (
          <Button type="primary" onClick={onAction}>
            {actionLabel}
          </Button>
        )}
      </Empty>
    </div>
  );
};
