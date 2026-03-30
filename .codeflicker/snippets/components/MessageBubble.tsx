/**
 * 可复用组件 - MessageBubble
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies react, antd, @ant-design/icons
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/chat/MessageBubble.tsx
 * 评分: 6.06
 * 复杂度: 8
 */

import React from 'react';
import { Avatar } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';
import { SourcePanel } from './SourcePanel';
import type { Message } from '@/types/api';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: isUser ? 'row-reverse' : 'row',
        gap: 12,
        marginBottom: 16,
        alignItems: 'flex-start',
      }}
    >
      <Avatar
        icon={isUser ? <UserOutlined /> : <RobotOutlined />}
        style={{ background: isUser ? '#1677ff' : '#52c41a', flexShrink: 0 }}
      />
      <div style={{ maxWidth: '70%' }}>
        <div
          style={{
            background: isUser ? '#1677ff' : '#fff',
            color: isUser ? '#fff' : '#333',
            padding: '10px 16px',
            borderRadius: isUser ? '12px 2px 12px 12px' : '2px 12px 12px 12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
            lineHeight: 1.6,
            whiteSpace: 'pre-wrap',
          }}
        >
          {message.content}
        </div>
        {!isUser && message.sources.length > 0 && (
          <SourcePanel sources={message.sources} />
        )}
      </div>
    </div>
  );
};
