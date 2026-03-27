# WebRTC（Web 实时通信）

## 1. 概述

WebRTC（Web Real-Time Communication）是 W3C 和 IETF 制定的开放标准，允许浏览器和移动应用之间建立**点对点（P2P）的实时音视频和数据通信**，无需安装插件或中间服务器转发媒体流。WebRTC 是目前唯一在浏览器端原生支持的全双工（Full-duplex）低延迟音视频通信标准。

**在 Tri-Transformer 中的角色**：Phase 3 数字人方案的核心通信基础设施，实现用户浏览器与 Tri-Transformer 推理服务之间的全双工音视频实时传输（延迟目标 < 300ms）。

---

## 2. 核心架构与协议栈

### 2.1 WebRTC 协议栈

```
应用层: JavaScript API (RTCPeerConnection, MediaStream)
        ↓
信令层: 应用自定义（WebSocket/HTTP/SIP）
        ↓
NAT穿透: ICE + STUN + TURN
        ↓
安全层: DTLS（数据加密）+ SRTP（媒体加密）
        ↓
传输层: UDP（主）/ TCP（备选）
        ↓
编解码: Opus（音频）+ VP8/VP9/H.264/AV1（视频）
```

### 2.2 建立连接流程（信令与 ICE）

```
用户浏览器 (Caller)              Tri-Transformer 服务端 (Callee)
                                 
1. createOffer()                 
   ↓ SDP Offer                  
2. ──────── 通过信令服务器发送 ──────→
                                 3. createAnswer()
4. ←──────── SDP Answer ─────────
   
5. 各自收集 ICE Candidates（本机IP/端口、STUN候选、TURN中继候选）
6. 通过信令交换 ICE Candidates
7. ICE 连接性检查（Connectivity Checks）
8. 选择最优候选对（Candidate Pair）
9. 建立 DTLS 握手（安全层）
10. 开始媒体传输（SRTP）
```

### 2.3 STUN vs. TURN

| 类型 | 作用 | 适用场景 | 带宽消耗 |
|---|---|---|---|
| **STUN** | 获取公网 IP 和端口 | 对称 NAT 以外的大多数情况 | 极低（仅探测） |
| **TURN** | 通过中继服务器转发媒体 | 对称 NAT、企业防火墙 | 高（所有媒体经服务器） |
| **直连 P2P** | 无 | 同局域网、公网直连 | 零服务器成本 |

---

## 3. 音视频编解码

### 3.1 音频：Opus 编解码器

Opus 是 WebRTC 的默认音频编解码器，专为实时通信设计：

| 特性 | 参数 |
|---|---|
| 采样率 | 8-48 kHz（自适应） |
| 码率范围 | 6-510 kbps |
| 帧长度 | 2.5-60ms |
| 延迟 | 编码约 20ms（超低延迟模式 < 5ms） |
| 应用场景 | 语音（SILK 模式）+ 音乐（CELT 模式）自动切换 |

**与 EnCodec/SNAC 的关系**：Tri-Transformer 输出的 Codec Token 在服务端解码为 PCM → Opus 编码 → WebRTC 传输 → 浏览器 Opus 解码 → 播放。

### 3.2 视频编解码策略

| 编解码器 | 延迟 | 质量 | 浏览器支持 | 推荐场景 |
|---|---|---|---|---|
| VP9 | 中 | 高 | Chrome/Firefox/Edge | 通用（推荐）|
| H.264 | 低 | 高 | 全平台 | 移动端/iOS |
| AV1 | 高（软编）| 极高 | Chrome 90+ | 高质量低码率 |
| VP8 | 极低 | 中 | 全平台 | 极低延迟场景 |

---

## 4. 实现方法

### 4.1 前端（React + WebRTC）

```javascript
class TriTransformerWebRTCClient {
    constructor(signalingServerUrl) {
        this.ws = new WebSocket(signalingServerUrl);
        this.pc = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                {
                    urls: 'turn:your-turn-server.com:3478',
                    username: 'user',
                    credential: 'password'
                }
            ]
        });
        
        this.setupSignaling();
        this.setupPeerConnection();
    }

    async startCall() {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { 
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            },
            video: { width: 640, height: 480, frameRate: 15 }
        });

        stream.getTracks().forEach(track => {
            this.pc.addTrack(track, stream);
        });

        const offer = await this.pc.createOffer();
        await this.pc.setLocalDescription(offer);
        this.ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }));
    }

    setupPeerConnection() {
        this.pc.ontrack = (event) => {
            const remoteStream = event.streams[0];
            document.getElementById('remoteAudio').srcObject = remoteStream;
        };

        this.pc.onicecandidate = (event) => {
            if (event.candidate) {
                this.ws.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate
                }));
            }
        };

        this.pc.onconnectionstatechange = () => {
            console.log(`Connection state: ${this.pc.connectionState}`);
        };
    }

    setupSignaling() {
        this.ws.onmessage = async (message) => {
            const data = JSON.parse(message.data);
            if (data.type === 'answer') {
                await this.pc.setRemoteDescription(new RTCSessionDescription(data));
            } else if (data.type === 'ice-candidate') {
                await this.pc.addIceCandidate(new RTCIceCandidate(data.candidate));
            }
        };
    }
}
```

### 4.2 服务端（Python aiortc）

```python
import asyncio
import json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

class TriTransformerAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, tri_transformer_engine):
        super().__init__()
        self.engine = tri_transformer_engine
        self.audio_buffer = asyncio.Queue()

    async def recv(self):
        frame = await self.audio_buffer.get()
        return frame

    async def process_input(self, input_frame):
        output_audio = await self.engine.generate_audio(input_frame)
        await self.audio_buffer.put(output_audio)

pcs = set()

async def offer(request):
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    engine = TriTransformerEngine()
    output_track = TriTransformerAudioTrack(engine)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.ensure_future(process_user_audio(track, engine))
        pc.addTrack(output_track)

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def process_user_audio(track, engine):
    while True:
        frame = await track.recv()
        await engine.push_audio_frame(frame)

app = web.Application()
app.router.add_post("/offer", offer)
web.run_app(app, port=8080)
```

---

## 5. 延迟优化

### 5.1 端到端延迟分解

```
用户说话 → 麦克风采集(5ms) → Opus编码(20ms) → 网络传输(20-100ms)
         → 服务端接收(5ms) → 音频Token化(10ms) → I-Transformer(50ms)
         → C-Transformer(20ms) → O-Transformer生成(50ms)
         → 音频解码(10ms) → Opus编码(20ms) → 网络传输(20-100ms)
         → 浏览器解码播放(10ms)
总计: ~210-450ms（目标 < 300ms）
```

### 5.2 延迟优化策略

```javascript
const pc = new RTCPeerConnection({
    iceServers: [...],

    iceTransportPolicy: 'all',
    bundlePolicy: 'max-bundle',
    rtcpMuxPolicy: 'require'
});

const sender = pc.addTrack(audioTrack, stream);
const params = sender.getParameters();
params.encodings[0].maxBitrate = 24000;
params.encodings[0].priority = 'high';
sender.setParameters(params);
```

---

## 6. 最新进展（2024-2025）

### 6.1 WebCodecs API
- 浏览器原生视频帧解码 API，允许直接获取原始 YUV 帧，无需通过 `<video>` 标签，为 Tri-Transformer 的实时视频分析提供更低延迟的帧获取路径。

### 6.2 WebTransport
- 基于 QUIC 的新一代 Web 传输协议，提供比 WebSocket 更低延迟、支持多路复用的双向流，是 WebRTC 信令通道的未来替代方案。

### 6.3 GPT-4o Realtime API
- OpenAI 基于 WebRTC 构建了 GPT-4o 的实时语音对话 API，是 Tri-Transformer 实时服务接口设计的直接参考。

### 6.4 WHIP/WHEP（WebRTC HTTP 媒体协议）
- 标准化 WebRTC 流媒体推流（WHIP）和拉流（WHEP）协议，简化 WebRTC 与 CDN/流媒体服务的集成，为 Tri-Transformer 的多用户并发流媒体服务提供标准化方案。
