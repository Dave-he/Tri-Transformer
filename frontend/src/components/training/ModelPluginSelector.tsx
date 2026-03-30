import React from 'react';
import { Input, Space, Typography } from 'antd';
import type { AvailableModel } from '@/types/trainingConfig';

const { Text } = Typography;

interface ModelPluginSelectorProps {
  availableModels: AvailableModel[];
  iModelId: string;
  oModelId: string;
  onIModelChange: (value: string) => void;
  onOModelChange: (value: string) => void;
}

export const ModelPluginSelector: React.FC<ModelPluginSelectorProps> = ({
  availableModels: _availableModels,
  iModelId,
  oModelId,
  onIModelChange,
  onOModelChange,
}) => {
  return (
    <div style={{ padding: '0 0 16px' }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <div>
          <Text strong>I-Transformer 底座模型（输入侧）</Text>
          <Input
            value={iModelId}
            onChange={(e) => onIModelChange(e.target.value)}
            placeholder="例如：Qwen/Qwen2-Audio-7B"
            style={{ marginTop: 8 }}
          />
        </div>
        <div>
          <Text strong>O-Transformer 底座模型（输出侧）</Text>
          <Input
            value={oModelId}
            onChange={(e) => onOModelChange(e.target.value)}
            placeholder="例如：meta-llama/Llama-3-8B"
            style={{ marginTop: 8 }}
          />
        </div>
      </Space>
    </div>
  );
};
