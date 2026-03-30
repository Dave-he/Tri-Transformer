/**
 * 可复用组件 - MessageList
 * 
 * @category components
 * @tags components, react, typescript
 * 
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/chat/MessageList.tsx
 * 评分: 5.75
 * 复杂度: 4
 */

import React, { useEffect, useRef } from 'react';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { EmptyState } from '@/components/common/EmptyState';
import { MessageBubble } from './MessageBubble';
import type { Message } from '@/types/api';

interface MessageListProps {
  messages: Message[];
  loading: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, loading }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0 && !loading) {
    return <EmptyState description="暂无消息，开始对话吧~" />;
  }

  return (
    <div style={{ flex: 1, overflow: 'auto', padding: '16px 24px' }}>
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {loading && <LoadingSpinner size="small" tip="正在生成回复..." />}
      <div ref={bottomRef} />
    </div>
  );
};
