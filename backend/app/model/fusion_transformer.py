"""
Tri-Transformer 紧耦合分组多头融合架构

核心设计原则：
- 最小改造兼容：基于标准 DiT Block（AdaLN-Zero 版）改造
- 掩码完全隔离：三组注意力头独立并行计算，掩码严格隔离
- 空间完全统一：所有模块共享隐层维度、嵌入层、位置编码
- 全范式兼容：同一套权重，支持正向 AR、反向 AR、扩散去噪三大生成范式
- 效率等价：推理延迟与同参数量标准 DiT/Transformer 完全一致
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerationMode(Enum):
    FORWARD_AR = "forward_ar"
    BACKWARD_AR = "backward_ar"
    DIFFUSION = "diffusion"


@dataclass
class FusionBlockConfig:
    d_model: int = 512
    num_heads_total: int = 12
    intermediate_size: int = 1536
    num_layers: int = 6
    dropout: float = 0.1
    rope_theta: float = 1_000_000.0
    max_len: int = 32768
    vocab_size: int = 32000
    use_gated_fusion: bool = False


class ModeEmbedding(nn.Module):
    def __init__(self, d_model: int, num_modes: int = 3):
        super().__init__()
        self.embeddings = nn.Embedding(num_modes, d_model)

    def forward(self, mode: GenerationMode) -> torch.Tensor:
        mode_idx = [
            GenerationMode.FORWARD_AR,
            GenerationMode.BACKWARD_AR,
            GenerationMode.DIFFUSION,
        ].index(mode)
        return self.embeddings(torch.tensor(mode_idx, device=self.embeddings.weight.device))


class TimeStepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.d_model = d_model

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.d_model // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        emb = self.mlp(emb)
        return emb


class InputFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mode_embed = ModeEmbedding(d_model)
        self.time_embed = TimeStepEmbedding(d_model)
        self.null_cond = nn.Parameter(torch.randn(1, d_model) * 0.02)
        nn.init.zeros_(self.null_cond)

    def forward(
        self,
        x_core: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        mode: GenerationMode = GenerationMode.DIFFUSION,
    ) -> torch.Tensor:
        B, S, _ = x_core.shape
        c_total = torch.zeros(B, self.d_model, device=x_core.device, dtype=x_core.dtype)

        if timesteps is not None:
            c_total = c_total + self.time_embed(timesteps)
        else:
            ar_timestep = torch.zeros(B, device=x_core.device, dtype=torch.long)
            c_total = c_total + self.time_embed(ar_timestep)

        if cond is not None:
            c_total = c_total + cond
        else:
            c_total = c_total + self.null_cond

        mode_emb = self.mode_embed(mode)
        c_total = c_total + mode_emb

        c_broadcast = c_total.unsqueeze(1)
        return x_core + c_broadcast


class RoPEFusion(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half_fusion(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_fusion(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half_fusion(q) * sin
    k_rot = k * cos + rotate_half_fusion(k) * sin
    return q_rot, k_rot


class GroupedMaskMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads_total: int,
        dropout: float = 0.0,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        assert d_model % num_heads_total == 0
        assert num_heads_total % 3 == 0

        self.d_model = d_model
        self.num_heads_total = num_heads_total
        self.head_dim = d_model // num_heads_total
        self.h_group = num_heads_total // 3

        self.q_proj_f = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)
        self.k_proj_f = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)
        self.v_proj_f = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)

        self.q_proj_b = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)
        self.k_proj_b = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)
        self.v_proj_b = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)

        self.q_proj_d = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)
        self.k_proj_d = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)
        self.v_proj_d = nn.Linear(d_model, self.h_group * self.head_dim, bias=False)

        self.o_proj = nn.Linear(num_heads_total * self.head_dim, d_model, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.rope = RoPEFusion(self.head_dim, rope_theta)
        self.attn_drop = nn.Dropout(dropout)

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        return torch.matmul(attn_weights, v)

    def _project_and_rope(
        self,
        x: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = q_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
        k = k_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
        v = v_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rope(T, x.device)
        q, k = apply_rotary_fusion(q, k, cos, sin)

        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_mask: Optional[torch.Tensor] = None,
        backward_mask: Optional[torch.Tensor] = None,
        bidirectional_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q_f, k_f, v_f = self._project_and_rope(
            hidden_states, self.q_proj_f, self.k_proj_f, self.v_proj_f
        )
        q_b, k_b, v_b = self._project_and_rope(
            hidden_states, self.q_proj_b, self.k_proj_b, self.v_proj_b
        )
        q_d, k_d, v_d = self._project_and_rope(
            hidden_states, self.q_proj_d, self.k_proj_d, self.v_proj_d
        )

        attn_f = self._compute_attention(q_f, k_f, v_f, forward_mask)
        attn_b = self._compute_attention(q_b, k_b, v_b, backward_mask)
        attn_d = self._compute_attention(q_d, k_d, v_d, bidirectional_mask)

        attn_f = attn_f.transpose(1, 2).contiguous().view(B, T, self.h_group * self.head_dim)
        attn_b = attn_b.transpose(1, 2).contiguous().view(B, T, self.h_group * self.head_dim)
        attn_d = attn_d.transpose(1, 2).contiguous().view(B, T, self.h_group * self.head_dim)

        attn_concat = torch.cat([attn_f, attn_b, attn_d], dim=-1)
        return self.o_proj(attn_concat)


class GatedFusionAttention(nn.Module):
    def __init__(self, d_model: int, num_heads_total: int, dropout: float = 0.0, rope_theta: float = 1_000_000.0):
        super().__init__()
        self.gmmha = GroupedMaskMultiHeadAttention(d_model, num_heads_total, dropout, rope_theta)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_mask: Optional[torch.Tensor] = None,
        backward_mask: Optional[torch.Tensor] = None,
        bidirectional_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_output = self.gmmha(hidden_states, forward_mask, backward_mask, bidirectional_mask)

        if cond is not None:
            gate_input = torch.cat([hidden_states, cond], dim=-1)
        else:
            gate_input = hidden_states

        gates = self.gate_mlp(gate_input)
        g_f, g_b, g_d = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]
        g_f, g_b, g_d = torch.sigmoid(g_f), torch.sigmoid(g_b), torch.sigmoid(g_d)

        h_group = self.gmmha.h_group
        head_dim = self.gmmha.head_dim

        f_out = base_output[:, :, 0:h_group * head_dim] * g_f.unsqueeze(-1)
        b_out = base_output[:, :, h_group * head_dim:2 * h_group * head_dim] * g_b.unsqueeze(-1)
        d_out = base_output[:, :, 2 * h_group * head_dim:] * g_d.unsqueeze(-1)

        return f_out + b_out + d_out


class AdaLNZeroModulation(nn.Module):
    def __init__(self, d_model: int, separate_modulation: bool = False):
        super().__init__()
        self.separate_modulation = separate_modulation

        if separate_modulation:
            self.mlp_f = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            self.mlp_b = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            self.mlp_d = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            for mlp in [self.mlp_f, self.mlp_b, self.mlp_d]:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
        else:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c_total: torch.Tensor) -> dict:
        if self.separate_modulation:
            params_f = self.mlp_f(c_total)
            params_b = self.mlp_b(c_total)
            params_d = self.mlp_d(c_total)
            D = c_total.size(-1)
            return {
                "shift_a_f": params_f[:, :, 0:D],
                "scale_a_f": params_f[:, :, D:2*D],
                "gate_a_f": params_f[:, :, 2*D:3*D],
                "shift_m_f": params_f[:, :, 3*D:4*D],
                "scale_m_f": params_f[:, :, 4*D:5*D],
                "gate_m_f": params_f[:, :, 5*D:],
                "shift_a_b": params_b[:, :, 0:D],
                "scale_a_b": params_b[:, :, D:2*D],
                "gate_a_b": params_b[:, :, 2*D:3*D],
                "shift_m_b": params_b[:, :, 3*D:4*D],
                "scale_m_b": params_b[:, :, 4*D:5*D],
                "gate_m_b": params_b[:, :, 5*D:],
                "shift_a_d": params_d[:, :, 0:D],
                "scale_a_d": params_d[:, :, D:2*D],
                "gate_a_d": params_d[:, :, 2*D:3*D],
                "shift_m_d": params_d[:, :, 3*D:4*D],
                "scale_m_d": params_d[:, :, 4*D:5*D],
                "gate_m_d": params_d[:, :, 5*D:],
            }
        else:
            params = self.mlp(c_total)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params.chunk(6, dim=-1)
            return dict(
                shift_a=shift_a, scale_a=scale_a, gate_a=gate_a,
                shift_m=shift_m, scale_m=scale_m, gate_m=gate_m,
            )


class FusionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads_total: int,
        intermediate_size: int,
        dropout: float = 0.1,
        rope_theta: float = 1_000_000.0,
        use_gated_fusion: bool = False,
        separate_modulation: bool = False,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d_model)
        self.post_attention_layernorm = nn.RMSNorm(d_model)

        if use_gated_fusion:
            self.self_attn = GatedFusionAttention(
                d_model, num_heads_total, dropout, rope_theta
            )
        else:
            self.self_attn = GroupedMaskMultiHeadAttention(
                d_model, num_heads_total, dropout, rope_theta
            )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, d_model),
        )

        self.adaln = AdaLNZeroModulation(d_model, separate_modulation=separate_modulation)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        c_total: torch.Tensor,
        forward_mask: Optional[torch.Tensor] = None,
        backward_mask: Optional[torch.Tensor] = None,
        bidirectional_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        h = self.input_layernorm(x)

        adaln_params = self.adaln(c_total)

        if "shift_a_f" in adaln_params:
            h = h * (1 + adaln_params["scale_a_f"]) + adaln_params["shift_a_f"]
        else:
            h = h * (1 + adaln_params["scale_a"]) + adaln_params["shift_a"]

        attn_out = self.self_attn(h, forward_mask, backward_mask, bidirectional_mask)

        if "gate_a_f" in adaln_params:
            x = residual + adaln_params["gate_a_f"] * self.drop(attn_out)
        else:
            x = residual + adaln_params["gate_a"] * self.drop(attn_out)

        residual = x
        h = self.post_attention_layernorm(x)

        if "shift_m_f" in adaln_params:
            h = h * (1 + adaln_params["scale_m_f"]) + adaln_params["shift_m_f"]
        else:
            h = h * (1 + adaln_params["scale_m"]) + adaln_params["shift_m"]

        mlp_out = self.mlp(h)

        if "gate_m_f" in adaln_params:
            x = residual + adaln_params["gate_m_f"] * self.drop(mlp_out)
        else:
            x = residual + adaln_params["gate_m"] * self.drop(mlp_out)

        return x


class FusionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads_total: int = 12,
        num_layers: int = 6,
        intermediate_size: int = 1536,
        dropout: float = 0.1,
        rope_theta: float = 1_000_000.0,
        max_len: int = 32768,
        use_gated_fusion: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.input_fusion = InputFusion(d_model)

        self.layers = nn.ModuleList([
            FusionBlock(
                d_model=d_model,
                num_heads_total=num_heads_total,
                intermediate_size=intermediate_size,
                dropout=dropout,
                rope_theta=rope_theta,
                use_gated_fusion=use_gated_fusion,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(d_model)

        self.forward_ar_head = nn.Linear(d_model, vocab_size, bias=False)
        self.backward_ar_head = nn.Linear(d_model, vocab_size, bias=False)
        self.diffusion_head = nn.Linear(d_model, d_model, bias=False)

    def create_masks(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_f = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask_b = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask_d = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

        return mask_f, mask_b, mask_d

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        mode: GenerationMode = GenerationMode.DIFFUSION,
        target: Optional[torch.Tensor] = None,
    ) -> dict:
        x = self.embedding(input_ids) * (self.d_model ** 0.5)

        x_fused = self.input_fusion(x, timesteps, cond, mode)

        seq_len = x.size(1)
        mask_f, mask_b, mask_d = self.create_masks(seq_len, x.device, x.dtype)

        for layer in self.layers:
            x_fused = layer(x_fused, x_fused, mask_f, mask_b, mask_d)

        hidden = self.norm(x_fused)

        outputs = {
            "hidden": hidden,
            "forward_ar_logits": self.forward_ar_head(hidden),
            "backward_ar_logits": self.backward_ar_head(hidden),
            "diffusion_features": self.diffusion_head(hidden),
        }

        if target is not None:
            outputs["forward_ar_loss"] = F.cross_entropy(
                outputs["forward_ar_logits"].transpose(-2, -1), target
            )
            outputs["backward_ar_loss"] = F.cross_entropy(
                outputs["backward_ar_logits"].transpose(-2, -1), target
            )

        return outputs

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = (
            self.parameters() if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)
