# Qwen3 架构（基于 Qwen3 实现 Tri-Transformer）

## 0. 结论先行

- **核心架构特点**：GQA（num_kv_heads=8）+ RoPE（rope_theta=1M）+ RMSNorm + QK-Norm + SwiGLU MLP，词表 152K，原生支持 32K 上下文（可扩展至 128K），是当前最强开源 Dense LLM。
- **Tri-Transformer 最优选型**：I/O 端插拔大模型选 Qwen3-8B（理解/生成均衡，单 A100 80G 可跑缝合训练）；算力受限选 Qwen3-0.6B/1.7B 作为 I/O 端 + Qwen3-8B 作为 C-Transformer；MoE 路线选 Qwen3-30B-A3B（激活参数 3B，显存友好）。
- **工程推荐**：直接用 HuggingFace `transformers>=4.51.0` 加载；非思考模式（设置 `enable_thinking=False`）用于实时对话；LoRA 微调参数：`r=64, alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`。
- **关键参数与 C-Transformer 对齐**：`hidden_size=4096`（8B），`num_attention_heads=32`，`num_key_value_heads=8`，`head_dim=128`——C-Transformer 设计维度应与此对齐以消除接缝。

---

## 1. 概述

Qwen3 是阿里云通义团队于 2025 年发布的第三代 Qwen 系列大语言模型，提供 Dense（0.6B/1.7B/4B/8B/14B/32B）和 MoE（30B-A3B / 235B-A22B）两条产品线，全系支持 **128K 上下文**（8B+），并引入**思考模式（Thinking Mode）**与**非思考模式（Non-Thinking Mode）**的动态切换。

Qwen3 在架构上沿用并升级了 Qwen2.5 的 GQA + RoPE + RMSNorm 体系，新增 **QK-Norm**（逐头归一化）、超大 `rope_theta=1,000,000`（支持极长上下文），以及 MoE 变体中的细粒度专家路由。

**在 Tri-Transformer 中的角色**：
- **I-Transformer**：加载 Qwen3-8B/14B Dense 权重作为流式输入编码骨干（左端插拔大模型 A），利用其强大的多语言理解与长上下文能力。
- **O-Transformer**：加载 Qwen3-8B/32B Dense 权重作为输出解码骨干（右端插拔大模型 B）。
- **C-Transformer**：基于 Qwen3 Block 的超参，设计与两端权重维度兼容的 DiT 控制中枢。
- **MoE 路线**：Qwen3-30B-A3B 可作为单 GPU（40G）全流程训练的混合专家骨干，激活参数仅 3B。

---

## 2. Qwen3 完整架构参数

### 2.1 Dense 系列超参

| 模型 | 层数 | hidden_size | Q Heads | KV Heads | FFN inter | vocab | ctx | rope_θ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 1024 | 16 | 8 | 3072 | 151936 | 32K | 1,000,000 |
| Qwen3-1.7B | 28 | 2048 | 16 | 8 | 8192 | 151936 | 32K | 1,000,000 |
| Qwen3-4B | 36 | 2560 | 32 | 8 | 10240 | 151936 | 128K | 1,000,000 |
| **Qwen3-8B** | **36** | **4096** | **32** | **8** | **12288** | **151936** | **128K** | **1,000,000** |
| Qwen3-14B | 40 | 5120 | 40 | 8 | 17920 | 151936 | 128K | 1,000,000 |
| Qwen3-32B | 64 | 5120 | 64 | 8 | 25600 | 151936 | 128K | 1,000,000 |

### 2.2 MoE 系列超参

| 模型 | 层数 | hidden_size | Q Heads | KV Heads | 专家数 | 每次激活 | 专家FFN | 激活参数 | ctx |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 48 | 2048 | 32 | 4 | 128 | 8 | 768 | ~3B | 128K |
| **Qwen3-235B-A22B** | **94** | **4096** | **64** | **4** | **128** | **8** | **1536** | **~22B** | **128K** |

---

## 3. 关键技术组件详解

### 3.1 QK-Norm（查询-键逐头归一化）

Qwen3 新增的稳定性关键特性：对 Q 和 K **逐注意力头**独立做 RMSNorm，防止多模态混合训练时的梯度爆炸。

```python
import torch
import torch.nn as nn
import math

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(hidden_states.dtype)


class Qwen3Attention(nn.Module):
    """
    Qwen3 注意力层：GQA + QK-Norm + RoPE（rope_theta=1,000,000）
    """
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 131072,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(head_dim)
        self.k_norm = Qwen3RMSNorm(head_dim)

        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                    position_ids: torch.Tensor) -> tuple:
        seq_len = q.size(2)
        t = position_ids.float().unsqueeze(-1)
        freqs = t * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos[:, :, :k.size(2)] + rotate_half(k) * sin[:, :, :k.size(2)]
        return q_rot, k_rot

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_value: tuple = None,
        use_cache: bool = False,
    ) -> tuple:
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self._apply_rope(q, k, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        if use_cache:
            new_cache = (k, v)
        else:
            new_cache = None

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_cache
```

### 3.2 Qwen3 FFN（SwiGLU）

```python
class Qwen3MLP(nn.Module):
    """Qwen3 使用 SwiGLU 激活的 FFN（与 Llama3 一致）"""
    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 12288):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )
```

### 3.3 完整 Qwen3 Decoder Block

```python
class Qwen3DecoderLayer(nn.Module):
    """Qwen3 标准 Decoder Block：Pre-RMSNorm + GQA + QK-Norm + SwiGLU"""
    def __init__(self, hidden_size: int = 4096, num_heads: int = 32,
                 num_kv_heads: int = 8, intermediate_size: int = 12288):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden_size)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        self.mlp = Qwen3MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                past_key_value: tuple = None, use_cache: bool = False) -> tuple:
        residual = hidden_states
        hidden_states, new_cache = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_cache
```

### 3.4 Qwen3 MoE 专家层

```python
class Qwen3MoEMLP(nn.Module):
    """单个 MoE 专家的 FFN"""
    def __init__(self, hidden_size: int = 4096, moe_intermediate_size: int = 1536):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(moe_intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class Qwen3MoESparseMLP(nn.Module):
    """
    Qwen3 MoE 稀疏 FFN：128 专家，每次激活 8 个
    用于 Qwen3-30B-A3B（激活参数 ~3B）和 Qwen3-235B-A22B（激活参数 ~22B）
    """
    def __init__(
        self,
        hidden_size: int = 4096,
        moe_intermediate_size: int = 1536,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Qwen3MoEMLP(hidden_size, moe_intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden_states.shape
        hidden_flat = hidden_states.view(-1, D)

        router_logits = self.gate(hidden_flat)
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(hidden_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_ids == i).any(dim=-1)
            if not mask.any():
                continue
            expert_input = hidden_flat[mask]
            expert_out = expert(expert_input)
            idx = (topk_ids[mask] == i).float()
            weight = (topk_weights[mask] * idx).sum(dim=-1, keepdim=True)
            out[mask] += weight * expert_out

        return out.view(B, T, D)
```

---

## 4. Thinking Mode 实现（推理时动态切换）

Qwen3 独特的**双模式推理**：`enable_thinking=True` 触发内部 Chain-of-Thought（`<think>...</think>` 标签包裹），`enable_thinking=False` 直接生成答案，适合延迟敏感场景。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

def qwen3_generate(
    prompt: str,
    enable_thinking: bool = True,
    max_new_tokens: int = 512,
) -> tuple[str, str]:
    """
    返回 (thinking_content, final_answer)
    enable_thinking=True  → 触发 CoT，适合复杂推理
    enable_thinking=False → 直接回答，适合实时对话（低延迟）
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            top_k=20,
            do_sample=True,
        )

    new_tokens = output_ids[0][len(inputs.input_ids[0]):]
    full_output = tokenizer.decode(new_tokens, skip_special_tokens=False)

    thinking, answer = "", full_output
    if "<think>" in full_output and "</think>" in full_output:
        think_start = full_output.index("<think>") + len("<think>")
        think_end = full_output.index("</think>")
        thinking = full_output[think_start:think_end].strip()
        answer = full_output[think_end + len("</think>"):].strip()

    return thinking, answer


thinking, answer = qwen3_generate(
    "Tri-Transformer 的 C-Transformer 与标准 DiT 有什么本质区别？",
    enable_thinking=True,
)
print(f"思考过程:\n{thinking[:200]}...\n")
print(f"最终回答:\n{answer}")
```

---

## 5. 基于 Qwen3 的 Tri-Transformer 插拔方案

### 5.1 I-Transformer：加载 Qwen3 作为流式输入编码器

```python
from transformers import Qwen3Model, Qwen3Config
import torch.nn as nn

class ITransformerWithQwen3(nn.Module):
    """
    I-Transformer：Qwen3 Dense 作为 Streaming Decoder 骨干
    - 使用因果掩码（causal）处理实时流式输入
    - 通过 KV Cache 实现 0 延迟增量推理
    - 支持 LoRA 适配 C-Transformer 的控制信号注入
    """
    def __init__(
        self,
        qwen3_model_name: str = "Qwen/Qwen3-8B",
        d_ctrl: int = 4096,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.backbone = Qwen3Model.from_pretrained(
            qwen3_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        hidden_size = self.backbone.config.hidden_size
        self.ctrl_proj = nn.Linear(d_ctrl, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, d_ctrl, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        ctrl_signal: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = True,
    ) -> tuple:
        outputs = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden = outputs.last_hidden_state

        if ctrl_signal is not None:
            ctrl_bias = self.ctrl_proj(ctrl_signal).unsqueeze(1)
            hidden = hidden + ctrl_bias

        i_enc = self.out_proj(hidden)
        return i_enc, outputs.past_key_values
```

### 5.2 adaLN-Zero C-Transformer（Qwen3 维度兼容）

```python
class CTransformerQwen3Compatible(nn.Module):
    """
    C-Transformer：基于 Qwen3-8B 维度（hidden=4096）设计
    DiT 风格控制中枢，通过 adaLN-Zero 向 I/O 注入控制信号
    """
    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        num_layers: int = 8,
        num_slots: int = 16,
    ):
        super().__init__()
        self.state_slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)

        self.cross_attn_i = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn_o = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.dit_layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=d_model * 3,
            )
            for _ in range(num_layers)
        ])

        self.norm = Qwen3RMSNorm(d_model)

        self.adaLN_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        nn.init.zeros_(self.adaLN_out[-1].weight)
        nn.init.zeros_(self.adaLN_out[-1].bias)

    def forward(
        self,
        i_enc: torch.Tensor,
        o_prev: torch.Tensor,
        position_ids: torch.Tensor = None,
    ) -> tuple:
        B = i_enc.size(0)
        s = self.state_slots.expand(B, -1, -1).clone()

        s = s + self.cross_attn_i(s, i_enc, i_enc)[0]
        s = s + self.cross_attn_o(s, o_prev, o_prev)[0]

        if position_ids is None:
            position_ids = torch.arange(s.size(1), device=s.device).unsqueeze(0)

        for layer in self.dit_layers:
            s, _ = layer(s, position_ids=position_ids)

        s = self.norm(s)
        ctrl_signal = s.mean(dim=1)

        params = self.adaLN_out(ctrl_signal)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params.chunk(6, dim=-1)
        ctrl_params = {
            "shift_a": shift_a, "scale_a": scale_a, "gate_a": gate_a,
            "shift_m": shift_m, "scale_m": scale_m, "gate_m": gate_m,
        }
        return ctrl_signal, ctrl_params
```

### 5.3 LoRA 缝合训练配置（Qwen3 专用）

```python
from peft import LoraConfig, get_peft_model, TaskType

QWEN3_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

def build_qwen3_lora_config(r: int = 16, lora_alpha: float = 32) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=QWEN3_LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def apply_lora_to_i_transformer(i_transformer: ITransformerWithQwen3) -> nn.Module:
    """
    阶段 2 LoRA 缝合训练：冻结 Qwen3 主干，仅训练 LoRA 适配层
    可训练参数量：约 17M（r=16，Qwen3-8B 全注意力层）
    """
    lora_cfg = build_qwen3_lora_config(r=16)
    i_transformer.backbone = get_peft_model(i_transformer.backbone, lora_cfg)
    i_transformer.backbone.print_trainable_parameters()
    return i_transformer
```

---

## 6. Qwen3 在 Tri-Transformer 各训练阶段的使用策略

### 6.1 分阶段训练方案

```
阶段 1 — C-Transformer 对齐预热（2-4 周）
  冻结: Qwen3-8B I-端全部参数（无 LoRA）
        Qwen3-8B O-端全部参数
  训练: C-Transformer（14层 DiT，d=4096）+ 衔接投影层
  数据: 多模态对话数据，使用 Non-Thinking Mode 输出作为监督信号
  目标: C-Transformer 学会从 i_enc 读取语义，向 o_plan 发送控制信号

阶段 2 — LoRA 联合微调（4-6 周）
  冻结: Qwen3-8B I/O 端主干权重
  训练: C-Transformer + LoRA（r=16）注入 I/O 两端 Attention
  数据: 可控对话数据（带情感/语速/风格标注）
  目标: adaLN-Zero 调制在 Qwen3 层激活中生效，实现风格实时切换

阶段 3 — Thinking Mode 集成（可选，2 周）
  解冻: I-端 Qwen3 底层 12 层（Layer 0-11）
  训练: 端到端，引入 enable_thinking=True 的长链推理数据
  目标: Tri-Transformer 在复杂知识问答时触发内部 CoT，
        在实时对话时使用 Non-Thinking Mode 保证 <300ms 延迟
```

### 6.2 显存估算（Qwen3-8B × 2 + C-Transformer）

| 方案 | 参数量 | 精度 | 显存/卡 | 最低配置 |
|---|---|---|---|---|
| 阶段1（冻结I/O）| ~1B（C-Transformer）| BF16 | ~8GB | 单卡 A100 40G |
| 阶段2（LoRA）| ~1.03B（C + LoRA 17M×2）| BF16 | ~22GB | 单卡 A100 40G |
| 阶段3（局部解冻）| ~3.5B 可训练 | BF16 | ~48GB | 双卡 A100 40G |
| 全量微调 | ~17B（I+C+O）| BF16 + ZeRO-3 | ~20GB/卡 | 8卡 A100 40G |

---

## 7. 推理部署（vLLM + Qwen3）

### 7.1 vLLM 部署 Qwen3-8B

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    --enable-chunked-prefill \
    --trust-remote-code \
    --port 8000
```

### 7.2 流式推理（Non-Thinking Mode，适合实时对话）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "system", "content": "你是 Tri-Transformer 实时对话助手。"},
        {"role": "user", "content": "请简要介绍 adaLN-Zero 机制"},
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=True,
    temperature=0.7,
    max_tokens=256,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 7.3 Qwen3-30B-A3B MoE 单卡部署（激活参数仅 3B）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
```

> **注意**：30B-A3B 总参数 30B，加载需要 ~60GB 内存，但每次推理仅激活 ~3B 参数，推理速度约等于 3B Dense 模型。适合单 A100 80G 部署 Tri-Transformer 全系统（I + C + O 合计）。

---

## 8. 与 Qwen2.5 的关键改进对比

| 特性 | Qwen2.5 | Qwen3 |
|---|---|---|
| QK-Norm | 无 | **有**（每头 RMSNorm，稳定训练）|
| Thinking Mode | 无 | **有**（动态 CoT 开关）|
| rope_theta | 1,000,000 | 1,000,000（一致）|
| MoE 路线 | Qwen2-57B-A14B | **Qwen3-30B-A3B / 235B-A22B** |
| 最大上下文 | 128K | **128K**（8B+均支持）|
| 训练数据规模 | ~18T tokens | **~36T tokens** |
| 后训练 | SFT + DPO | **四阶段**（冷启动→推理RL→融合→通用RL）|
| 多语言 | 29 种语言 | **119 种语言** |
| 对 Tri-Transformer 价值 | 基础骨干 | **更强推理 + CoT 可控 + MoE 灵活部署** |
