"""
Tri-Transformer 三分支实现（Qwen3 架构风格）

架构特性：
  - RoPE 位置编码（rope_theta=1,000,000，与 Qwen3 一致）
  - GQA（分组查询注意力，num_kv_heads < num_heads）
  - QK-Norm（每注意力头独立 RMSNorm，防多模态训练梯度爆炸）
  - SwiGLU FFN（gate_proj × silu + up_proj → down_proj）
  - Pre-RMSNorm（残差前归一化，更稳定）
  - adaLN-Zero（C-Transformer → I/O 无侵入调制，零初始化）
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

class Qwen3RMSNorm(nn.Module):
    """Per-tensor RMSNorm（用于 Pre-Norm 和 adaLN 调制）。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x.to(x.dtype)


class Qwen3RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码，rope_theta=1,000,000（Qwen3 默认）。"""

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将 RoPE cos/sin 应用到 q, k（形状 [B, H, T, head_dim]）。"""
    cos = cos[:q.size(2)].unsqueeze(0).unsqueeze(0)
    sin = sin[:q.size(2)].unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Qwen3 风格注意力（GQA + QK-Norm + RoPE）
# ---------------------------------------------------------------------------

class Qwen3Attention(nn.Module):
    """
    GQA + QK-Norm + RoPE 注意力层。

    参数
    ----
    d_model      : 隐藏维度
    num_heads    : Q 头数
    num_kv_heads : KV 头数（GQA，num_kv_heads ≤ num_heads，且整除）
    dropout      : 注意力 dropout
    rope_theta   : RoPE base（Qwen3 默认 1,000,000）
    causal       : 是否施加因果掩码（True=Decoder, False=Encoder）
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 1_000_000.0,
        causal: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim)
        self.k_norm = Qwen3RMSNorm(self.head_dim)

        self.rope = Qwen3RotaryEmbedding(self.head_dim, rope_theta)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_source: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        hidden_states : [B, Tq, D]  Query 序列
        kv_source     : [B, Ts, D]  KV 序列（None 时退化为自注意力）
        attention_mask: [B, 1, Tq, Ts]  加法掩码（-inf 表示遮掩）
        """
        B, Tq, _ = hidden_states.shape
        kv_src = kv_source if kv_source is not None else hidden_states
        Ts = kv_src.size(1)

        q = self.q_proj(hidden_states).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_src).view(B, Ts, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_src).view(B, Ts, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm（逐头归一化）
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE（仅自注意力时施加，交叉注意力 kv_source 已有位置信息）
        if kv_source is None:
            cos, sin = self.rope(Tq, hidden_states.device)
            q, k = apply_rotary(q, k, cos, sin)

        # KV Cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_cache = (k, v) if use_cache else None

        # GQA：扩展 KV 头至 Q 头数
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 因果掩码（仅自注意力且 causal=True）
        if self.causal and kv_source is None:
            causal_mask = torch.tril(
                torch.ones(Tq, k.size(2), device=hidden_states.device, dtype=torch.bool)
            )
            attn_w = attn_w.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_w = attn_w + attention_mask

        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.attn_drop(attn_w)

        out = torch.matmul(attn_w, v)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.o_proj(out), new_cache


# ---------------------------------------------------------------------------
# Qwen3 风格 FFN（SwiGLU）
# ---------------------------------------------------------------------------

class Qwen3SwiGLU(nn.Module):
    """SwiGLU FFN：down(silu(gate(x)) * up(x))。"""

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Qwen3 Decoder Block（Pre-RMSNorm + adaLN-Zero 可选）
# ---------------------------------------------------------------------------

class Qwen3DecoderBlock(nn.Module):
    """
    标准 Qwen3 Decoder Block（Pre-RMSNorm）。

    可选接收 adaLN-Zero 参数（来自 C-Transformer），对注意力和 FFN 分支
    分别施加 scale/shift/gate 调制，实现无侵入的实时控制。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        rope_theta: float = 1_000_000.0,
        causal: bool = True,
    ):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(d_model)
        self.post_attention_layernorm = Qwen3RMSNorm(d_model)
        self.self_attn = Qwen3Attention(
            d_model, num_heads, num_kv_heads,
            dropout=dropout, rope_theta=rope_theta, causal=causal,
        )
        self.mlp = Qwen3SwiGLU(d_model, intermediate_size)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
        adaln_params: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """
        x            : [B, T, D]
        adaln_params : dict with keys shift_a, scale_a, gate_a, shift_m, scale_m, gate_m
                       shape of each: [B, D]  （由 C-Transformer 生成）
        """
        residual = x
        h = self.input_layernorm(x)

        if adaln_params is not None:
            scale_a = adaln_params["scale_a"].unsqueeze(1)
            shift_a = adaln_params["shift_a"].unsqueeze(1)
            h = h * (1 + scale_a) + shift_a

        attn_out, new_cache = self.self_attn(
            h, attention_mask=attention_mask,
            past_key_value=past_key_value, use_cache=use_cache,
        )

        if adaln_params is not None:
            gate_a = adaln_params["gate_a"].unsqueeze(1)
            x = residual + gate_a * self.drop(attn_out)
        else:
            x = residual + self.drop(attn_out)

        residual = x
        h = self.post_attention_layernorm(x)

        if adaln_params is not None:
            scale_m = adaln_params["scale_m"].unsqueeze(1)
            shift_m = adaln_params["shift_m"].unsqueeze(1)
            h = h * (1 + scale_m) + shift_m

        mlp_out = self.mlp(h)

        if adaln_params is not None:
            gate_m = adaln_params["gate_m"].unsqueeze(1)
            x = residual + gate_m * self.drop(mlp_out)
        else:
            x = residual + self.drop(mlp_out)

        return x, new_cache


# ---------------------------------------------------------------------------
# I-Transformer（Qwen3 风格 Causal Decoder → Bidirectional Encoder）
# ---------------------------------------------------------------------------

class ITransformer(nn.Module):
    """
    输入编码器：Qwen3 风格流式因果 Decoder 骨干。

    将 Token 序列编码为上下文感知隐状态（i_enc），供 C-Transformer 消费。
    支持 KV Cache 流式推理；接受来自 C-Transformer 的 adaLN-Zero 控制信号。

    骨干架构：Pre-RMSNorm + GQA + QK-Norm + RoPE(θ=1M) + SwiGLU
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        num_layers: int = 6,
        intermediate_size: int = 12288,
        dropout: float = 0.1,
        rope_theta: float = 1_000_000.0,
        max_len: int = 32768,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList([
            Qwen3DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                rope_theta=rope_theta,
                causal=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = Qwen3RMSNorm(d_model)

        # adaLN-Zero 投影：将 C-Transformer 控制信号映射为每层 6 个参数
        self.adaln_proj = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            for _ in range(num_layers)
        ])
        for proj in self.adaln_proj:
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

    def _make_adaln_params(self, ctrl_signal: torch.Tensor, layer_idx: int) -> dict:
        params = self.adaln_proj[layer_idx](ctrl_signal)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params.chunk(6, dim=-1)
        return dict(
            shift_a=shift_a, scale_a=scale_a, gate_a=gate_a,
            shift_m=shift_m, scale_m=scale_m, gate_m=gate_m,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        control_signal: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list]]:
        """
        src               : [B, T]  输入 token ids
        control_signal    : [B, D]  C-Transformer 全局控制向量
        past_key_values   : list of (k, v) per layer，KV Cache
        返回 (i_enc [B, T, D], new_past_key_values)
        """
        x = self.embedding(src) * math.sqrt(self.d_model)

        attention_mask = None
        if src_key_padding_mask is not None:
            attention_mask = src_key_padding_mask.float()
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(x.dtype).min

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            adaln = (
                self._make_adaln_params(control_signal, i)
                if control_signal is not None
                else None
            )
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = layer(
                x,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                adaln_params=adaln,
            )
            if use_cache:
                new_past_key_values.append(new_kv)

        return self.norm(x), new_past_key_values


# ---------------------------------------------------------------------------
# C-Transformer（DiT 风格控制中枢，State Slots + adaLN-Zero）
# ---------------------------------------------------------------------------

class CTransformer(nn.Module):
    """
    控制中枢：DiT 风格，维护全局状态槽（State Slots），通过交叉注意力
    感知 I 端编码与 O 端反馈，生成 adaLN-Zero 控制参数注入 I/O 两端。

    骨干同样使用 Qwen3 Decoder Block（无因果掩码），保证维度兼容性。
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        num_layers: int = 8,
        intermediate_size: int = 12288,
        num_slots: int = 16,
        dropout: float = 0.1,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots

        self.state_slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)

        self.cross_attn_i = Qwen3Attention(
            d_model, num_heads, num_kv_heads, dropout=dropout, causal=False
        )
        self.cross_attn_o = Qwen3Attention(
            d_model, num_heads, num_kv_heads, dropout=dropout, causal=False
        )
        self.cross_norm_i = Qwen3RMSNorm(d_model)
        self.cross_norm_o = Qwen3RMSNorm(d_model)

        self.dit_layers = nn.ModuleList([
            Qwen3DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                rope_theta=rope_theta,
                causal=False,
            )
            for _ in range(num_layers)
        ])
        self.norm = Qwen3RMSNorm(d_model)

        # 生成 I 端 adaLN-Zero 参数（每层）
        self.adaln_i_proj = nn.Linear(d_model, 6 * d_model)
        # 生成 O 端 adaLN-Zero 参数（每层，延迟到 O-Transformer 消费）
        self.adaln_o_proj = nn.Linear(d_model, 6 * d_model)

        nn.init.zeros_(self.adaln_i_proj.weight)
        nn.init.zeros_(self.adaln_i_proj.bias)
        nn.init.zeros_(self.adaln_o_proj.weight)
        nn.init.zeros_(self.adaln_o_proj.bias)

    def _split_adaln(self, proj: nn.Linear, signal: torch.Tensor) -> dict:
        params = proj(signal)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params.chunk(6, dim=-1)
        return dict(
            shift_a=shift_a, scale_a=scale_a, gate_a=gate_a,
            shift_m=shift_m, scale_m=scale_m, gate_m=gate_m,
        )

    def forward(
        self,
        i_enc: torch.Tensor,
        o_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict, dict]:
        """
        i_enc  : [B, Ti, D]  I-Transformer 输出编码
        o_prev : [B, To, D]  O-Transformer 上一步规划状态（None 时用零填充）
        返回
        -------
        ctrl_signal  : [B, D]    全局控制向量（均值池化）
        adaln_i      : dict      I 端 adaLN-Zero 参数
        adaln_o      : dict      O 端 adaLN-Zero 参数
        """
        B = i_enc.size(0)

        if o_prev is None:
            o_prev = torch.zeros(B, 1, self.d_model, device=i_enc.device, dtype=i_enc.dtype)

        s = self.state_slots.expand(B, -1, -1).clone()

        s_i, _ = self.cross_attn_i(self.cross_norm_i(s), kv_source=i_enc)
        s = s + s_i

        s_o, _ = self.cross_attn_o(self.cross_norm_o(s), kv_source=o_prev)
        s = s + s_o

        for layer in self.dit_layers:
            s, _ = layer(s)

        s = self.norm(s)
        ctrl_signal = s.mean(dim=1)

        adaln_i = self._split_adaln(self.adaln_i_proj, ctrl_signal)
        adaln_o = self._split_adaln(self.adaln_o_proj, ctrl_signal)

        return ctrl_signal, adaln_i, adaln_o


# ---------------------------------------------------------------------------
# O-Transformer（Qwen3 风格 Planning Encoder + Streaming Decoder）
# ---------------------------------------------------------------------------

class OTransformer(nn.Module):
    """
    输出解码器：Qwen3 风格受控自回归生成。

    Planning Encoder  : 双向注意力，融合 C-Transformer 控制信号与 i_enc 知识
    Streaming Decoder : 因果注意力，自回归生成 Token
    两阶段均接受 adaLN-Zero 控制参数（来自 C-Transformer）。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        num_plan_layers: int = 4,
        num_dec_layers: int = 6,
        intermediate_size: int = 12288,
        dropout: float = 0.1,
        rope_theta: float = 1_000_000.0,
        max_len: int = 32768,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Planning Encoder（双向，融合 i_enc + ctrl）
        self.plan_layers = nn.ModuleList([
            Qwen3DecoderBlock(
                d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size, dropout=dropout,
                rope_theta=rope_theta, causal=False,
            )
            for _ in range(num_plan_layers)
        ])
        self.plan_cross_attn = nn.ModuleList([
            Qwen3Attention(d_model, num_heads, num_kv_heads, dropout=dropout, causal=False)
            for _ in range(num_plan_layers)
        ])
        self.plan_cross_norm = nn.ModuleList([
            Qwen3RMSNorm(d_model) for _ in range(num_plan_layers)
        ])

        # Streaming Decoder（因果，自回归生成）
        self.dec_layers = nn.ModuleList([
            Qwen3DecoderBlock(
                d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size, dropout=dropout,
                rope_theta=rope_theta, causal=True,
            )
            for _ in range(num_dec_layers)
        ])
        self.dec_cross_attn = nn.ModuleList([
            Qwen3Attention(d_model, num_heads, num_kv_heads, dropout=dropout, causal=False)
            for _ in range(num_dec_layers)
        ])
        self.dec_cross_norm = nn.ModuleList([
            Qwen3RMSNorm(d_model) for _ in range(num_dec_layers)
        ])

        # adaLN-Zero 投影（接收来自 C-Transformer 的 adaln_o）
        self.adaln_plan_proj = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            for _ in range(num_plan_layers)
        ])
        self.adaln_dec_proj = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            for _ in range(num_dec_layers)
        ])
        for proj_list in [self.adaln_plan_proj, self.adaln_dec_proj]:
            for proj in proj_list:
                nn.init.zeros_(proj[-1].weight)
                nn.init.zeros_(proj[-1].bias)

        self.norm = Qwen3RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def _make_adaln(self, proj_list: nn.ModuleList, ctrl_signal: torch.Tensor,
                    layer_idx: int) -> Optional[dict]:
        if ctrl_signal is None:
            return None
        params = proj_list[layer_idx](ctrl_signal)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params.chunk(6, dim=-1)
        return dict(
            shift_a=shift_a, scale_a=scale_a, gate_a=gate_a,
            shift_m=shift_m, scale_m=scale_m, gate_m=gate_m,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        control_signal: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list]]:
        """
        tgt            : [B, T]   目标 token ids
        memory         : [B, Ti, D]  i_enc（I-Transformer 编码）
        control_signal : [B, D]   C-Transformer 控制向量

        返回 (logits [B, T, vocab], o_hidden [B, T, D], new_past_key_values)
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)

        new_past = [] if use_cache else None

        # --- Planning Encoder ---
        for i, (plan_layer, cross_attn, cross_norm) in enumerate(
            zip(self.plan_layers, self.plan_cross_attn, self.plan_cross_norm)
        ):
            adaln = self._make_adaln(self.adaln_plan_proj, control_signal, i)
            x, kv = plan_layer(x, adaln_params=adaln, use_cache=use_cache)
            if use_cache:
                new_past.append(kv)
            # 交叉注意力融合 i_enc
            x_cross, _ = cross_attn(cross_norm(x), kv_source=memory)
            x = x + x_cross

        # --- Streaming Decoder ---
        for i, (dec_layer, cross_attn, cross_norm) in enumerate(
            zip(self.dec_layers, self.dec_cross_attn, self.dec_cross_norm)
        ):
            adaln = self._make_adaln(self.adaln_dec_proj, control_signal, i)
            past_kv = (
                past_key_values[len(self.plan_layers) + i]
                if past_key_values is not None
                else None
            )
            x, kv = dec_layer(
                x, adaln_params=adaln,
                past_key_value=past_kv, use_cache=use_cache,
            )
            if use_cache:
                new_past.append(kv)
            x_cross, _ = cross_attn(cross_norm(x), kv_source=memory)
            x = x + x_cross

        o_hidden = self.norm(x)
        logits = self.output_proj(o_hidden)
        return logits, o_hidden, new_past
