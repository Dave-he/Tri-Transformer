# BPE（字节对编码 / Byte Pair Encoding）

## 1. 概述

BPE（Byte Pair Encoding）最初是一种数据压缩算法，由 Sennrich et al. 于 2016 年引入 NLP 领域（arXiv:1508.07909，ACL 2016），用于解决神经机器翻译中的开放词汇表（Open Vocabulary）问题。通过迭代合并最高频字符对构建子词词表，BPE 已成为 GPT、Llama、Qwen、Claude 等所有主流 Transformer 语言模型的标准文本 Tokenizer。

**在 Tri-Transformer 中的角色**：文本模态的标准 Token 化方案，与音频 Codec Token、视觉 VQ Token 共同构成统一的多模态离散 Token 空间。

---

## 2. 算法原理

### 2.1 BPE 训练算法

```
输入: 语料库字符序列（每词末尾添加 </w> 标记）
初始词表: 所有字符（Unicode）

循环直到达到目标词表大小 V：
    1. 统计所有相邻符号对（bigram）的出现频次
    2. 选出频次最高的符号对 (a, b)
    3. 将语料中所有 "a b" 合并为 "ab"
    4. 将 "ab" 添加到词表
    5. 更新合并规则列表

输出: 合并规则列表 + 最终词表
```

### 2.2 算法示例

```
语料: {"low": 5, "lower": 2, "newest": 6, "widest": 3}
初始字符: l-o-w-</w>, l-o-w-e-r-</w>, n-e-w-e-s-t-</w>, w-i-d-e-s-t-</w>

第1轮: 最高频对 = ("e","s") 频次=9 → 合并为 "es"
       l-o-w-</w>, l-o-w-e-r-</w>, n-e-w-es-t-</w>, w-i-d-es-t-</w>

第2轮: 最高频对 = ("es","t") 频次=9 → 合并为 "est"
       ...n-e-w-est-</w>, w-i-d-est-</w>

第3轮: 最高频对 = ("est","</w>") 频次=9 → 合并为 "est</w>"
       ... → 获得子词 "est"
```

### 2.3 Python 实现

```python
from collections import Counter, defaultdict
import re

def get_vocab(text: str) -> dict:
    """将文本转为字符级词频字典"""
    words = text.lower().split()
    return Counter(' '.join(list(w)) + ' </w>' for w in words)

def get_stats(vocab: dict) -> Counter:
    """统计所有相邻符号对的频次"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair: tuple, vocab: dict) -> dict:
    """将最高频对合并"""
    out = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        out[pattern.sub(''.join(pair), word)] = vocab[word]
    return out

def train_bpe(text: str, num_merges: int) -> list:
    vocab = get_vocab(text)
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
    return merges
```

---

## 3. BPE 的现代变体

### 3.1 字节级 BPE（Byte-level BPE，GPT-2/GPT-3/Llama）

将 BPE 作用在原始 **字节（Byte）** 而非 Unicode 字符上：

```
优点:
- 词表大小固定为 256（字节数）+ 合并规则
- 无 <UNK>（任何文本均可编码）
- 多语言友好（中文、阿拉伯文等无需特殊处理）

GPT-4 tiktoken 词表大小: 100,277
Llama 3 词表大小: 128,256（含特殊 Token）
```

### 3.2 WordPiece（BERT 的 Tokenizer）

类似 BPE，但贪婪选择**最大化训练数据对数似然**的合并，而非频次最高的对。使用 `##` 标记非词首子词。

### 3.3 SentencePiece（多语言 Tokenizer）

- 直接作用于原始字符流（包括空格），无需预分词。
- 同时支持 BPE 和 Unigram LM 两种算法。
- Google T5、XLM-R、Qwen 系列使用 SentencePiece BPE。

---

## 4. 使用方法

### 4.1 使用 tiktoken（OpenAI，适用于 GPT/Llama 3）

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Tri-Transformer 多模态系统"
tokens = enc.encode(text)
print(f"Token IDs: {tokens}")
print(f"Token 数量: {len(tokens)}")
print(f"Token 字符串: {[enc.decode([t]) for t in tokens]}")

text_back = enc.decode(tokens)
```

### 4.2 使用 tokenizers 库（HuggingFace）

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=10000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.train(files=["data.txt"], trainer=trainer)
tokenizer.save("bpe_tokenizer.json")

output = tokenizer.encode("Hello, world!")
print(output.tokens)
print(output.ids)
```

### 4.3 在 Tri-Transformer 中的多模态 Token 空间扩展

```python
from transformers import AutoTokenizer

base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

AUDIO_TOKEN_OFFSET = base_tokenizer.vocab_size
VISUAL_TOKEN_OFFSET = AUDIO_TOKEN_OFFSET + 4096 * 8
MODALITY_TOKENS = {
    "<|audio_start|>": AUDIO_TOKEN_OFFSET - 4,
    "<|audio_end|>": AUDIO_TOKEN_OFFSET - 3,
    "<|visual_start|>": AUDIO_TOKEN_OFFSET - 2,
    "<|visual_end|>": AUDIO_TOKEN_OFFSET - 1,
}

base_tokenizer.add_special_tokens({
    "additional_special_tokens": list(MODALITY_TOKENS.keys())
})
```

---

## 5. 最新进展（2024-2025）

### 5.1 Tiktoken 与 GPT-4 词表
- GPT-4（cl100k_base）：100,277 个 Token，中文字符约 3-4 Token/字（相比 GPT-3.5 提升约 2-3×）。
- Llama 3（128K 词表）：中文友好，单字一般为 1-2 Token。

### 5.2 超大词表趋势
- Gemma 2（Google, 2024）：256K 词表。
- 超大词表减少序列长度（尤其对中文、代码），但增大嵌入层参数量，需权衡。

### 5.3 Codec Token 与 BPE 的统一
- AnyGPT、Chameleon 等均采用将 Codec/VQ Token 直接追加到 BPE 词表的策略。
- 实践中，BPE 文本词表（约 32K-128K）+ Codec Token 词表（约 4K-32K）合并后，使用**统一嵌入矩阵**，通过模态标识符区分。

### 5.4 MegaByte 的字节级建模
- 彻底抛弃 BPE，以原始字节为基本单元，用小模型处理字节级序列，大模型处理 Patch 级序列，实现无限词汇表。尚处研究阶段。
