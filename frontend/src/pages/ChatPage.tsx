import React, { useEffect } from 'react';
import { Layout } from 'antd';
import { ConversationList } from '@/components/chat/ConversationList';
import { MessageList } from '@/components/chat/MessageList';
import { ChatInput } from '@/components/chat/ChatInput';
import { useConversation } from '@/hooks/useConversation';

const { Sider, Content } = Layout;

const ChatPage: React.FC = () => {
  const {
    conversations,
    currentConversationId,
    messages,
    loading,
    fetchConversations,
    createConversation,
    setActiveConversation,
    sendMessage,
    exportHistory,
  } = useConversation();

  useEffect(() => {
    fetchConversations().catch(() => undefined);
  }, [fetchConversations]);

  const handleSend = async (content: string) => {
    if (!currentConversationId) {
      await createConversation();
    }
    await sendMessage(content);
  };

  return (
    <Layout style={{ height: 'calc(100vh - 112px)', background: '#fff', borderRadius: 8 }}>
      <Sider width={260} theme="light" style={{ borderRight: '1px solid #f0f0f0', overflow: 'auto' }}>
        <ConversationList
          conversations={conversations}
          activeId={currentConversationId}
          onSelect={setActiveConversation}
          onCreate={createConversation}
          onExport={() => exportHistory('markdown')}
        />
      </Sider>
      <Layout style={{ display: 'flex', flexDirection: 'column' }}>
        <Content style={{ flex: 1, overflow: 'auto', background: '#fafafa' }}>
          <MessageList messages={messages} loading={loading} />
        </Content>
        <ChatInput onSend={handleSend} loading={loading} />
      </Layout>
    </Layout>
  );
};

export default ChatPage;
