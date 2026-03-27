from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from app.model.branches import ITransformer, CTransformer, OTransformer


@dataclass
class TriTransformerConfig:
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    num_layers_i: int = 6
    num_layers_c: int = 4
    num_layers_o: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_len: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class TriTransformerOutput:
    logits: torch.Tensor
    i_hidden: torch.Tensor
    control_signal: torch.Tensor
    o_hidden: torch.Tensor


class TriTransformerModel(nn.Module):
    """
    Tri-Transformer 三分支扭合模型。

    前向传播流程：
      src (input ids)  ──►  ITransformer  ──► i_enc
                                               │
                                               ▼
      i_enc ──► CTransformer ──► ctrl_signal ──┐
                    ▲                          │
                    │ (可选 o_prev 反馈)         │
                    │                          ▼
      tgt (decoder ids) ──► OTransformer (memory=i_enc, ctrl=ctrl_signal)
                                               │
                                               ▼
                                           logits (B, tgt_len, vocab_size)
    """

    def __init__(self, config: TriTransformerConfig):
        super().__init__()
        self.config = config

        self.i_branch = ITransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers_i,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_len=config.max_len,
        )
        self.c_branch = CTransformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers_c,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_len=config.max_len,
        )
        self.o_branch = OTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers_o,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
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
        i_enc = self.i_branch(src, src_key_padding_mask=src_key_padding_mask)
        ctrl_signal = self.c_branch(i_enc, o_prev=o_prev)
        logits, o_hidden = self.o_branch(
            tgt,
            memory=i_enc,
            control_signal=ctrl_signal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return TriTransformerOutput(
            logits=logits,
            i_hidden=i_enc,
            control_signal=ctrl_signal,
            o_hidden=o_hidden,
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)
