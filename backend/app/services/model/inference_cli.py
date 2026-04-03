"""
Tri-Transformer 模型推理脚本

支持：
- 单句推理
- 对话模式
- 流式输出
- 批量推理
"""
import argparse
import torch
from typing import Optional, List, Generator


class TriTransformerInference:
    """Tri-Transformer 推理引擎"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.config = None
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        from app.model.tri_transformer import TriTransformerModel, TriTransformerConfig
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        config_dict = checkpoint.get("config", {})
        if "model_config" in config_dict:
            self.config = TriTransformerConfig(**config_dict["model_config"])
        else:
            self.config = TriTransformerConfig()
        
        self.model = TriTransformerModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        try:
            from app.model.tokenizer.text_tokenizer import TextTokenizer
            self.tokenizer = TextTokenizer()
        except ImportError:
            self.tokenizer = None
            print("⚠️  TextTokenizer 未找到，使用基础分词")
        
        print(f"✓ 模型已加载：{self.checkpoint_path}")
        print(f"  设备：{self.device}")
        print(f"  词表大小：{self.config.vocab_size:,}")
    
    def encode(self, text: str, max_length: int = 64) -> torch.Tensor:
        """编码文本"""
        if self.tokenizer:
            ids = self.tokenizer.encode(text, max_length=max_length, truncation=True)
        else:
            ids = [ord(c) % self.config.vocab_size for c in text[:max_length]]
            ids = ids + [0] * (max_length - len(ids))
        
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def decode(self, token_ids: List[int]) -> str:
        """解码 token ids"""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        else:
            return "".join(chr(min(max(t, 32), 126)) for t in token_ids)
    
    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        context: Optional[str] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        生成文本
        
        参数:
            input_text: 输入文本
            context: 上下文文本（可选）
            max_length: 最大生成长度
            temperature: 温度（>1 增加随机性，<1 减少随机性）
            top_k: Top-k 采样
            top_p: Nucleus 采样
            num_return_sequences: 返回序列数
        
        返回:
            生成的文本列表
        """
        full_input = f"{context} {input_text}".strip() if context else input_text
        input_ids = self.encode(full_input, max_length=64)
        
        B = input_ids.size(0)
        src = input_ids
        tgt = input_ids[:, :32]
        
        output = self.model(src, tgt)
        logits = output.logits
        
        if temperature != 1.0:
            logits = logits / temperature
        
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        generated_sequences = []
        for _ in range(num_return_sequences):
            next_token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
            next_token = next_token.view(B, -1)
            
            generated_ids = next_token[0].tolist()
            generated_text = self.decode(generated_ids)
            generated_sequences.append(generated_text)
        
        return generated_sequences
    
    @torch.no_grad()
    def chat(
        self,
        message: str,
        history: Optional[List[dict]] = None,
        max_length: int = 128,
    ) -> str:
        """
        对话模式
        
        参数:
            message: 用户消息
            history: 对话历史
            max_length: 最大生成长度
        
        返回:
            模型回复
        """
        context = ""
        if history:
            context = " ".join([
                f"{h['role']}: {h['content']}"
                for h in history[-5:]
            ])
        
        generated = self.generate(
            input_text=message,
            context=context,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        
        return generated[0]
    
    @torch.no_grad()
    def stream_generate(
        self,
        input_text: str,
        context: Optional[str] = None,
        max_length: int = 128,
    ) -> Generator[str, None, None]:
        """
        流式生成（逐 token 输出）
        
        参数:
            input_text: 输入文本
            context: 上下文文本
            max_length: 最大生成长度
        
        返回:
            生成的 token 流
        """
        full_input = f"{context} {input_text}".strip() if context else input_text
        input_ids = self.encode(full_input, max_length=64)
        
        src = input_ids
        tgt = input_ids[:, :32]
        
        output = self.model(src, tgt)
        logits = output.logits
        
        generated_ids = []
        for position in range(min(max_length, logits.size(1))):
            next_token_logits = logits[:, position, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            token_id = next_token[0].item()
            
            generated_ids.append(token_id)
            
            if token_id == self.config.eos_token_id:
                break
            
            yield self.decode([token_id])


def interactive_mode(inference: TriTransformerInference):
    """交互式对话模式"""
    print("\n" + "="*60)
    print("Tri-Transformer 交互式对话".center(60))
    print("="*60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("="*60 + "\n")
    
    history = []
    
    while True:
        try:
            user_input = input("👤 您：").strip()
            
            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 再见！")
                break
            
            if user_input.lower() == "clear":
                history = []
                print("✓ 对话历史已清空\n")
                continue
            
            if not user_input:
                continue
            
            response = inference.chat(
                message=user_input,
                history=history,
                max_length=128,
            )
            
            print(f"🤖 AI: {response}\n")
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误：{e}\n")


def main():
    parser = argparse.ArgumentParser(description="Tri-Transformer 推理")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "single", "batch"],
        help="推理模式",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="单句推理输入文本",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="上下文文本",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="最大生成长度",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1,
        help="生成序列数",
    )
    
    args = parser.parse_args()
    
    inference = TriTransformerInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    if args.mode == "interactive":
        interactive_mode(inference)
    
    elif args.mode == "single":
        if not args.input:
            print("❌ 单句模式需要提供 --input 参数")
            return
        
        generated = inference.generate(
            input_text=args.input,
            context=args.context,
            max_length=args.max_length,
            temperature=args.temperature,
            num_return_sequences=args.num_sequences,
        )
        
        print(f"\n输入：{args.input}")
        if args.context:
            print(f"上下文：{args.context}")
        print(f"\n生成结果:")
        for i, text in enumerate(generated, 1):
            print(f"  [{i}] {text}")
    
    elif args.mode == "batch":
        inputs = [
            "你好，请介绍一下自己",
            "什么是人工智能？",
            "如何学习编程？",
        ]
        
        print("\n批量推理:")
        for input_text in inputs:
            generated = inference.generate(
                input_text=input_text,
                max_length=args.max_length,
                temperature=0.7,
                num_return_sequences=1,
            )
            print(f"\n输入：{input_text}")
            print(f"生成：{generated[0]}")


if __name__ == "__main__":
    main()
