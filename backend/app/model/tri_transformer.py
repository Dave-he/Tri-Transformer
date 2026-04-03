from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from app.model.branches import ITransformer, CTransformer, OTransformer


@dataclass
class TriTransformerConfig:
    """
    Tri-Transformer 配置（Qwen3 架构风格）。

    默认值对应"轻量研究规格"（d_model=512），可按 Qwen3-8B 规格
    （d_model=4096, num_heads=32, num_kv_heads=8）扩展。
    """
    # 词表
    vocab_size: int = 151936          # 对齐 Qwen3 词表大小
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # 模型维度（轻量研究规格）
    d_model: int = 512
    num_heads: int = 8                # Q 头数
    num_kv_heads: int = 2             # KV 头数（GQA，需整除 num_heads）
    intermediate_size: int = 1536     # SwiGLU FFN 中间维度（约 3× d_model）

    # 各分支层数
    num_layers_i: int = 6             # I-Transformer Decoder 层数
    num_layers_c: int = 4             # C-Transformer DiT 层数
    num_slots_c: int = 16             # C-Transformer State Slots 数量
    num_plan_layers_o: int = 3        # O-Transformer Planning Encoder 层数
    num_dec_layers_o: int = 6         # O-Transformer Streaming Decoder 层数

    # 训练正则化
    dropout: float = 0.1

    # RoPE
    rope_theta: float = 1_000_000.0  # Qwen3 默认：1,000,000

    # 上下文长度
    max_len: int = 32768


# Qwen3-8B 规格（生产插拔，需配合 LoRA 缝合训练）
QWEN3_8B_CONFIG = TriTransformerConfig(
    vocab_size=151936,
    d_model=4096,
    num_heads=32,
    num_kv_heads=8,
    intermediate_size=12288,
    num_layers_i=6,
    num_layers_c=8,
    num_slots_c=16,
    num_plan_layers_o=4,
    num_dec_layers_o=6,
    dropout=0.0,
    rope_theta=1_000_000.0,
    max_len=32768,
)

# Qwen3-30B-A3B MoE 兼容规格（单卡全系统）
QWEN3_30B_CONFIG = TriTransformerConfig(
    vocab_size=151936,
    d_model=2048,
    num_heads=32,
    num_kv_heads=4,
    intermediate_size=6144,
    num_layers_i=6,
    num_layers_c=6,
    num_slots_c=16,
    num_plan_layers_o=3,
    num_dec_layers_o=6,
    dropout=0.0,
    rope_theta=1_000_000.0,
    max_len=32768,
)


@dataclass
class TriTransformerOutput:
    logits: torch.Tensor
    i_hidden: torch.Tensor
    ctrl_signal: torch.Tensor
    o_hidden: torch.Tensor
    adaln_i: dict
    adaln_o: dict


class TriTransformerModel(nn.Module):
    """
    Tri-Transformer 三分支扭合模型（Qwen3 架构风格）。

    前向传播流程
    ------------
    src ──► ITransformer(Qwen3 Causal Decoder) ──► i_enc
                                                      │
                                                      ▼
                  i_enc ──► CTransformer(DiT+Slots) ──► ctrl_signal, adaln_i, adaln_o
                                ▲
                                │ o_prev 反馈
                                │
    tgt ──► OTransformer(Planning Enc + Streaming Dec) ◄── ctrl_signal(adaln_o), i_enc
                                                      │
                                                      ▼
                                          logits [B, tgt_len, vocab_size]

    关键特性
    --------
    - RoPE (rope_theta=1M)  : 支持长上下文（原生 32K，YaRN 扩展至 128K）
    - GQA                   : num_kv_heads < num_heads，降低 KV Cache 显存
    - QK-Norm               : 每注意力头独立 RMSNorm，防止多模态训练梯度失配
    - SwiGLU FFN            : silu(gate) * up → down
    - adaLN-Zero            : C-Transformer → I/O 的零初始化无侵入调制
    - Pre-RMSNorm           : 训练稳定性优于 Post-LN
    """

    def __init__(self, config: TriTransformerConfig):
        super().__init__()
        self.config = config

        self.i_branch = ITransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers_i,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            max_len=config.max_len,
        )
        self.c_branch = CTransformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers_c,
            intermediate_size=config.intermediate_size,
            num_slots=config.num_slots_c,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
        )
        self.o_branch = OTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            num_plan_layers=config.num_plan_layers_o,
            num_dec_layers=config.num_dec_layers_o,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            max_len=config.max_len,
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        o_prev: Optional[torch.Tensor] = None,
    ) -> TriTransformerOutput:
        # I-Transformer（首次前向无控制信号）
        i_enc, _ = self.i_branch(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )

        # C-Transformer → 生成控制信号与 adaLN-Zero 参数
        ctrl_signal, adaln_i, adaln_o = self.c_branch(i_enc, o_prev=o_prev)

        # I-Transformer 二次前向（注入控制信号）
        i_enc, _ = self.i_branch(
            src,
            src_key_padding_mask=src_key_padding_mask,
            control_signal=ctrl_signal,
        )

        # O-Transformer
        logits, o_hidden, _ = self.o_branch(
            tgt,
            memory=i_enc,
            control_signal=ctrl_signal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return TriTransformerOutput(
            logits=logits,
            i_hidden=i_enc,
            ctrl_signal=ctrl_signal,
            o_hidden=o_hidden,
            adaln_i=adaln_i,
            adaln_o=adaln_o,
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = (
            self.parameters() if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)
