import React from 'react';
import { Collapse, Tag } from 'antd';
import { LinkOutlined } from '@ant-design/icons';
import type { MessageSource } from '@/types/api';

interface SourcePanelProps {
  sources: MessageSource[];
}

export const SourcePanel: React.FC<SourcePanelProps> = ({ sources }) => {
  if (sources.length === 0) return null;

  return (
    <Collapse
      size="small"
      style={{ marginTop: 8 }}
      items={[
        {
          key: '1',
          label: (
            <span>
              <LinkOutlined style={{ marginRight: 4 }} />
              知识来源（{sources.length}）
            </span>
          ),
          children: sources.map((src, i) => (
            <div key={i} style={{ padding: '4px 0', borderBottom: i < sources.length - 1 ? '1px solid #f0f0f0' : 'none' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 500, color: '#1677ff' }}>{src.document}</span>
                <Tag color="blue">相关度 {(src.score * 100).toFixed(0)}%</Tag>
              </div>
              <div style={{ color: '#666', fontSize: 12, marginTop: 4, lineClamp: 2 }}>{src.chunk}</div>
            </div>
          )),
        },
      ]}
    />
  );
};
