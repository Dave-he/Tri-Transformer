import React, { useState, useRef } from 'react';
import { Button, Input } from 'antd';
import { SendOutlined } from '@ant-design/icons';

const { TextArea } = Input;

interface ChatInputProps {
  onSend: (content: string) => void;
  loading: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, loading }) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || loading) return;
    onSend(trimmed);
    setValue('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      style={{
        padding: '12px 16px',
        background: '#fff',
        borderTop: '1px solid #f0f0f0',
        display: 'flex',
        gap: 8,
        alignItems: 'flex-end',
      }}
    >
      <TextArea
        ref={textareaRef as React.Ref<HTMLTextAreaElement>}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="输入消息，按 Enter 发送，Shift+Enter 换行..."
        autoSize={{ minRows: 1, maxRows: 5 }}
        style={{ flex: 1, resize: 'none' }}
        disabled={loading}
      />
      <Button
        type="primary"
        icon={<SendOutlined />}
        onClick={handleSend}
        disabled={loading || !value.trim()}
        style={{ height: 36 }}
      >
        发送
      </Button>
    </div>
  );
};
