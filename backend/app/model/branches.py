import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ITransformer(nn.Module):
    """输入编码器：将 token ids 编码为上下文感知的隐状态序列。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        control_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        if control_signal is not None:
            x = x + control_signal
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class CTransformer(nn.Module):
    """控制中枢：通过双向交叉注意力监控 I/O 状态，输出全局控制信号。"""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_attn_i_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_attn_o_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
            for _ in range(num_layers)
        ])
        self.state_slot = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.state_slot, std=0.02)

    def forward(
        self,
        i_enc: torch.Tensor,
        o_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = i_enc.size(0)
        ctrl = self.state_slot.expand(B, -1, -1)

        if o_prev is None:
            o_prev = torch.zeros_like(i_enc)

        for idx in range(len(self.self_attn_layers)):
            n0, n1, n2, n3 = self.norm_layers[idx]
            ctrl_norm = n0(ctrl)
            sa_out, _ = self.self_attn_layers[idx](ctrl_norm, ctrl_norm, ctrl_norm)
            ctrl = ctrl + sa_out

            ctrl_norm = n1(ctrl)
            ca_i_out, _ = self.cross_attn_i_layers[idx](ctrl_norm, i_enc, i_enc)
            ctrl = ctrl + ca_i_out

            ctrl_norm = n2(ctrl)
            ca_o_out, _ = self.cross_attn_o_layers[idx](ctrl_norm, o_prev, o_prev)
            ctrl = ctrl + ca_o_out

            ctrl_norm = n3(ctrl)
            ctrl = ctrl + self.ffn_layers[idx](ctrl_norm)

        return ctrl


class OTransformer(nn.Module):
    """输出解码器：受控自回归生成，融合 I 分支编码与 C 分支控制信号。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.ctrl_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        control_signal: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        ctrl_bias = self.ctrl_proj(control_signal)
        memory_with_ctrl = torch.cat([memory, ctrl_bias], dim=1)

        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=tgt.device
            )

        x = self.decoder(
            x,
            memory_with_ctrl,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits, x
