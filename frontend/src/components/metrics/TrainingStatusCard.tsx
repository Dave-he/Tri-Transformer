import React from 'react';
import { Card, Progress, Descriptions } from 'antd';
import type { TrainingStatus } from '@/types/api';

interface TrainingStatusCardProps {
  status: TrainingStatus | null;
  loading?: boolean;
}

export const TrainingStatusCard: React.FC<TrainingStatusCardProps> = ({ status, loading }) => {
  if (!status) {
    return <Card loading={loading}><div>暂无训练状态</div></Card>;
  }

  return (
    <Card title="训练状态" loading={loading}>
      <Descriptions column={1} size="small">
        <Descriptions.Item label="当前阶段">{status.phase}</Descriptions.Item>
        <Descriptions.Item label="状态消息">{status.message}</Descriptions.Item>
        <Descriptions.Item label="预计完成">{status.eta}</Descriptions.Item>
      </Descriptions>
      <div style={{ marginTop: 16 }}>
        <div style={{ marginBottom: 8, color: '#666', fontSize: 12 }}>训练进度</div>
        <Progress percent={status.progress} status="active" />
      </div>
    </Card>
  );
};
