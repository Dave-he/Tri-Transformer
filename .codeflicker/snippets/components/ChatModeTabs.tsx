/**
 * 可复用组件 - ChatModeTabs
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies react, antd
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/chat/ChatModeTabs.tsx
 * 评分: 5.83
 * 复杂度: 5
 */

import React from 'react';
import { Tabs } from 'antd';

export type ChatMode = 'text' | 'audio' | 'video';

interface ChatModeTabsProps {
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  children: React.ReactNode;
  audioContent?: React.ReactNode;
  videoContent?: React.ReactNode;
}

const tabItems = [
  { key: 'text', label: '文本' },
  { key: 'audio', label: '语音' },
  { key: 'video', label: '视频' },
];

export const ChatModeTabs: React.FC<ChatModeTabsProps> = ({
  mode,
  onModeChange,
  children,
  audioContent,
  videoContent,
}) => {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Tabs
        activeKey={mode}
        onChange={(key) => onModeChange(key as ChatMode)}
        items={tabItems}
        style={{ flexShrink: 0, padding: '0 16px' }}
      />
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {mode === 'text' && children}
        {mode === 'audio' && (audioContent ?? null)}
        {mode === 'video' && (videoContent ?? null)}
      </div>
    </div>
  );
};
