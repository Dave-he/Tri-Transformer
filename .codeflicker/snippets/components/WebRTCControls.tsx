/**
 * 可复用组件 - WebRTCControls
 * 
 * @category components
 * @tags components, react, typescript, dependencies
 * @dependencies react, antd
 * 
 * 来源: /mnt/ssd/codespace/Tri-Transformer/frontend/src/components/chat/WebRTCControls.tsx
 * 评分: 5.68
 * 复杂度: 3
 */

interface WebRTCControlsProps {
  connectionState: WebRTCConnectionState;
  onStartCall: () => void;
  onEndCall: () => void;
  onInterrupt: () => void;
  onStyleChange: (value: number) => void;
}

const stateConfig: Record<WebRTCConnectionState, { color: string; text: string }> = {
  idle: { color: 'default', text: '未连接' },
  requesting_media: { color: 'processing', text: '获取设备...' },
  connecting: { color: 'processing', text: '连接中' },
  connected: { color: 'success', text: '通话中' },
  disconnected: { color: 'default', text: '已断开' },
  error: { color: 'error', text: '连接失败' },
};

export const WebRTCControls: React.FC<WebRTCControlsProps> = ({
  connectionState,
  onStartCall,
  onEndCall,
  onInterrupt,
  onStyleChange,
}) => {
  const { color, text } = stateConfig[connectionState];

  return (
    <div style={{ padding: '16px' }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Badge status={color as 'default' | 'processing' | 'success' | 'error' | 'warning'} text={text} />
        </Space>

        <Space>
          {connectionState === 'idle' || connectionState === 'disconnected' || connectionState === 'error' ? (
            <Button type="primary" onClick={onStartCall}>
              开始通话
            </Button>
          ) : connectionState === 'connected' ? (
            <>
              <Button danger onClick={onEndCall}>
                结束通话
              </Button>
              <Button onClick={onInterrupt}>
                打断
              </Button>
            </>
          ) : (
            <Button disabled loading>
              连接中
            </Button>
          )}
        </Space>

        {connectionState === 'connected' && (
          <div>
            <div style={{ marginBottom: 4, fontSize: 12, color: '#666' }}>正式度</div>
            <Slider
              min={0}
              max={100}
              defaultValue={50}
              onChange={onStyleChange}
              style={{ width: 200 }}
            />
  