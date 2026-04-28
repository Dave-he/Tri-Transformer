#!/usr/bin/env python3
"""
Tri-Transformer checkpoint → GGUF 转换脚本

仅转换 O-Transformer Streaming Decoder 分支。
I/C 分支结构无法适配 llama.cpp 标准 LLM 转换。

用法:
    python backend/scripts/convert_to_gguf.py --checkpoint path/to/checkpoint.pt --output model.gguf
    python backend/scripts/convert_to_gguf.py --checkpoint path/to/checkpoint.pt --quantize Q5_K_M
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Tri-Transformer → GGUF Converter")
    parser.add_argument("--checkpoint", required=True, help="Path to Tri-Transformer checkpoint (.pt)")
    parser.add_argument("--output", default=None, help="Output GGUF file path (default: <checkpoint>.F16.gguf)")
    parser.add_argument("--quantize", default=None,
                        choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
                        help="Quantization type (requires llama-quantize binary)")
    parser.add_argument("--llama-quantize-path", default=None,
                        help="Path to llama-quantize binary")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        base = os.path.splitext(checkpoint_path)[0]
        output_path = f"{base}.F16.gguf"

    print(f"🔄 转换 checkpoint: {checkpoint_path}")
    print(f"   输出: {output_path}")

    from app.model.gguf_converter import TriTransformerGGUFConverter

    converter = TriTransformerGGUFConverter(checkpoint_path=checkpoint_path)
    converter.convert_to_gguf(output_path=output_path)
    print(f"✅ GGUF F16 转换完成: {output_path}")

    if args.quantize:
        quantize_bin = args.llama_quantize_path
        if quantize_bin is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            quantize_bin = os.path.join(script_dir, "..", "llama.cpp-install", "bin", "llama-quantize")

        if not os.path.exists(quantize_bin):
            print(f"⚠️  llama-quantize not found: {quantize_bin}")
            print("   请先运行: bash backend/scripts/build_llamacpp_jetson.sh")
            sys.exit(1)

        quant_output = output_path.replace(".F16.gguf", f".{args.quantize}.gguf")
        print(f"🔄 量化 {args.quantize}: {output_path} → {quant_output}")
        os.system(f"{quantize_bin} {output_path} {quant_output} {args.quantize}")
        print(f"✅ 量化完成: {quant_output}")


if __name__ == "__main__":
    main()
