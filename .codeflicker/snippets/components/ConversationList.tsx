/**
 * 可复用组件 - ConversationList
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies react, antd, @ant-design/icons
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/chat/ConversationList.tsx
 * 评分: 5.68
 * 复杂度: 3
 */

interface ConversationListProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onExport: () => void;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  activeId,
  onSelect,
  onCreate,
  onExport,
}) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '12px 16px', display: 'flex', gap: 8 }}>
        <Button type="primary" icon={<PlusOutlined />} onClick={onCreate} block>
          新建对话
        </Button>
        {activeId && (
          <Tooltip title="导出当前对话">
            <Button icon={<DownloadOutlined />} onClick={onExport} />
          </Tooltip>
        )}
      </div>
      <List
        style={{ flex: 1, overflow: 'auto' }}
        dataSource={conversations}
        renderItem={(conv) => (
          <List.Item
            onClick={() => onSelect(conv.id)}
            style={{
              cursor: 'pointer',
              padding: '10px 16px',
              background: activeId === conv.id ? '#e6f4ff' : 'transparent',
              borderLeft: activeId === conv.id ? '3px solid #1677ff' : '3px solid transparent',
            }}
          >
            <List.Item.Meta
              title={<span style={{ fontSize: 13 }}>{conv.title}</span>}
              description={<span style={{ fontSize: 11, color: '#999' }}>{new Date(conv.updatedAt).toLocaleDateString()}</span>}
            />
          </List.Item>
        )}
      />
    </div>
  );
};
