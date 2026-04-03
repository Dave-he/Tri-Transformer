"""
Tri-Transformer 模型评估脚本

支持：
- 困惑度评估
- 准确率评估
- 幻觉检测评估
- RAG 检索评估
- 对话质量评估
"""
import argparse
import json
import torch
from pathlib import Path
from typing import Optional, List
from torch.utils.data import DataLoader, TensorDataset

from app.services.model.evaluation import (
    TriTransformerEvaluator,
    PerplexityEvaluator,
    AccuracyEvaluator,
    HallucinationEvaluator,
    RAGEvaluator,
    DialogEvaluator,
)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.config = None
        self.evaluator = None
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
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
        
        self.evaluator = TriTransformerEvaluator(self.config)
        
        print(f"✓ 模型已加载：{self.checkpoint_path}")
        print(f"  设备：{self.device}")
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        max_batches: int = 10,
    ) -> dict:
        """评估数据集"""
        all_metrics = {
            "perplexity": [],
            "accuracy": [],
            "hallucination_rate": [],
            "retrieval_precision": [],
            "retrieval_recall": [],
        }
        
        for i, batch in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            
            src, tgt_in, tgt_out = batch
            src = src.to(self.device)
            tgt_in = tgt_in.to(self.device)
            tgt_out = tgt_out.to(self.device)
            
            output = self.model(src, tgt_in)
            
            metrics = self.evaluator.evaluate(
                output,
                tgt_out,
            )
            
            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key].append(value)
        
        avg_metrics = {
            key: sum(values) / max(len(values), 1)
            for key, values in all_metrics.items()
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def evaluate_synthetic(self, num_samples: int = 100) -> dict:
        """使用合成数据评估"""
        vocab_size = self.config.vocab_size
        seq_len = 64
        batch_size = 8
        
        src = torch.randint(1, vocab_size, (num_samples, seq_len))
        tgt_in = torch.randint(1, vocab_size, (num_samples, seq_len))
        tgt_out = torch.randint(1, vocab_size, (num_samples, seq_len))
        
        dataset = TensorDataset(src, tgt_in, tgt_out)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return self.evaluate_dataset(loader, max_batches=None)
    
    def run_full_evaluation(
        self,
        test_loader: Optional[DataLoader] = None,
        num_synthetic_samples: int = 100,
    ) -> dict:
        """运行完整评估"""
        print("\n📊 运行完整评估...\n")
        
        metrics = {}
        
        if test_loader is not None:
            print("1️⃣  评估测试数据集...")
            test_metrics = self.evaluate_dataset(test_loader, max_batches=20)
            metrics["test"] = test_metrics
        else:
            print("1️⃣  评估合成数据...")
            synthetic_metrics = self.evaluate_synthetic(num_synthetic_samples)
            metrics["synthetic"] = synthetic_metrics
        
        print("\n评估完成！\n")
        
        return metrics


def create_test_dataset(
    vocab_size: int,
    num_samples: int = 200,
    seq_len: int = 64,
) -> DataLoader:
    """创建测试数据集"""
    src = torch.randint(1, vocab_size, (num_samples, seq_len))
    tgt_in = torch.randint(1, vocab_size, (num_samples, seq_len))
    tgt_out = torch.randint(1, vocab_size, (num_samples, seq_len))
    
    dataset = TensorDataset(src, tgt_in, tgt_out)
    return DataLoader(dataset, batch_size=8, shuffle=False)


def print_evaluation_report(metrics: dict):
    """打印评估报告"""
    print("\n" + "="*70)
    print("📊 评估报告".center(70))
    print("="*70)
    
    for dataset_name, dataset_metrics in metrics.items():
        print(f"\n{dataset_name.upper()} 数据集:")
        print("-" * 70)
        
        if "perplexity" in dataset_metrics:
            print(f"  Perplexity (困惑度):  {dataset_metrics['perplexity']:.2f}")
        
        if "accuracy" in dataset_metrics:
            print(f"  Accuracy (准确率):    {dataset_metrics['accuracy']:.4f}")
        
        if "hallucination_rate" in dataset_metrics:
            print(f"  Hallucination Rate:   {dataset_metrics['hallucination_rate']:.4f}")
        
        if "retrieval_precision" in dataset_metrics:
            print(f"  Retrieval Precision:  {dataset_metrics['retrieval_precision']:.4f}")
        
        if "retrieval_recall" in dataset_metrics:
            print(f"  Retrieval Recall:     {dataset_metrics['retrieval_recall']:.4f}")
    
    print("\n" + "="*70)
    print("\n评估指标说明:")
    print("  - Perplexity: 越低越好，表示模型预测更准确")
    print("  - Accuracy: 越高越好，表示 token 预测准确率")
    print("  - Hallucination Rate: 越低越好，表示幻觉内容更少")
    print("  - Retrieval Precision/Recall: 越高越好，表示检索质量更好")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tri-Transformer 模型评估")
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
        "--output",
        type=str,
        default="./evaluation_results.json",
        help="评估结果输出文件",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="评估样本数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小",
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    metrics = evaluator.run_full_evaluation(
        num_synthetic_samples=args.num_samples,
    )
    
    print_evaluation_report(metrics)
    
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ 评估结果已保存至：{output_path}")


if __name__ == "__main__":
    main()
