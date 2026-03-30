import React, { useEffect, useState } from 'react';
import { Layout } from 'antd';
import { ConversationList } from '@/components/chat/ConversationList';
import { MessageList } from '@/components/chat/MessageList';
import { ChatInput } from '@/components/chat/ChatInput';
import { ChatModeTabs } from '@/components/chat/ChatModeTabs';
import { WebRTCControls } from '@/components/chat/WebRTCControls';
import { AudioVisualizer } from '@/components/chat/AudioVisualizer';
import { useConversation } from '@/hooks/useConversation';
import { useWebRTCStore } from '@/store/webrtcStore';
import type { ChatMode } from '@/components/chat/ChatModeTabs';

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

  const [chatMode, setChatMode] = useState<ChatMode>('text');
  const { connectionState, localStream, remoteStream, startCall, endCall, sendInterrupt } = useWebRTCStore();

  useEffect(() => {
    fetchConversations().catch(() => undefined);
  }, [fetchConversations]);

  const handleSend = async (content: string) => {
    if (!currentConversationId) {
      await createConversation();
    }
    await sendMessage(content);
  };

  const textContent = (
    <Layout style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Content style={{ flex: 1, overflow: 'auto', background: '#fafafa' }}>
        <MessageList messages={messages} loading={loading} />
      </Content>
      <ChatInput onSend={handleSend} loading={loading} />
    </Layout>
  );

  const audioContent = (
    <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
      <AudioVisualizer stream={localStream ?? remoteStream} />
      <WebRTCControls
        connectionState={connectionState}
        onStartCall={() => startCall(false)}
        onEndCall={endCall}
        onInterrupt={() => void sendInterrupt()}
        onStyleChange={() => undefined}
      />
    </div>
  );

  const videoContent = (
    <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
      {remoteStream && (
        <video
          autoPlay
          playsInline
          ref={(el) => { if (el) el.srcObject = remoteStream; }}
          style={{ width: '100%', borderRadius: 8, background: '#000' }}
        />
      )}
      <WebRTCControls
        connectionState={connectionState}
        onStartCall={() => startCall(true)}
        onEndCall={endCall}
        onInterrupt={() => void sendInterrupt()}
        onStyleChange={() => undefined}
      />
    </div>
  );

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
        <ChatModeTabs
          mode={chatMode}
          onModeChange={setChatMode}
          audioContent={audioContent}
          videoContent={videoContent}
        >
          {textContent}
        </ChatModeTabs>
      </Layout>
    </Layout>
  );
};

export default ChatPage;
