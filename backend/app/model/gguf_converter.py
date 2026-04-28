import json
import os
import struct
import tempfile
from typing import Optional

import torch


def extract_o_branch_weights(model_state_dict: dict, prefix: str = "o_branch.streaming_decoder.") -> dict:
    extracted = {}
    for key, value in model_state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            extracted[new_key] = value
    return extracted


class GGUFConverter:
    """
    Convert Tri-Transformer O-Transformer Streaming Decoder weights to GGUF format.

    Only the O-Transformer Streaming Decoder branch is converted because:
    1. I/C branches have non-standard architectures (adaLN-Zero, State Slots) not supported by llama.cpp
    2. The Streaming Decoder is a standard causal decoder compatible with llama.cpp's LLM conversion pipeline
    """

    def __init__(self, checkpoint_path: Optional[str] = None, o_branch_state_dict: Optional[dict] = None):
        self.checkpoint_path = checkpoint_path
        self._o_branch_state_dict = o_branch_state_dict
        self._checkpoint_data = None

        if checkpoint_path is not None:
            self._checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

    def extract_o_branch_weights(self) -> dict:
        if self._o_branch_state_dict is not None:
            return self._o_branch_state_dict
        if self._checkpoint_data is None:
            raise ValueError("No checkpoint data loaded")
        model_state = self._checkpoint_data.get("model_state_dict", self._checkpoint_data)
        return extract_o_branch_weights(model_state)

    def _save_safetensors(self, weights: dict, output_dir: str) -> str:
        model_dir = os.path.join(output_dir, "hf_model")
        os.makedirs(model_dir, exist_ok=True)

        safetensors_path = os.path.join(model_dir, "model.safetensors")
        try:
            from safetensors.torch import save_file
            save_file(weights, safetensors_path)
        except ImportError:
            torch.save(weights, os.path.join(model_dir, "pytorch_model.bin"))

        config = {
            "architectures": ["TriTransformerOBranth"],
            "model_type": "tri_transformer_o_branch",
            "d_model": 512,
            "num_heads": 8,
            "num_kv_heads": 2,
            "vocab_size": 32000,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 32768,
            "intermediate_size": 1536,
            "hidden_act": "silu",
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        return model_dir

    def _run_hf_to_gguf(self, model_dir: str, output_dir: str, model_name: str) -> str:
        f16_gguf = os.path.join(output_dir, f"{model_name}_F16.gguf")
        convert_script = os.path.join(
            os.environ.get("LLAMACPP_PATH", "llama.cpp"), "convert_hf_to_gguf.py"
        )
        return f16_gguf

    def convert(self, output_dir: str, model_name: str = "tri_transformer_o") -> str:
        weights = self.extract_o_branch_weights()
        model_dir = self._save_safetensors(weights, output_dir)
        f16_gguf = self._run_hf_to_gguf(model_dir, output_dir, model_name)
        return f16_gguf

    def quantize(self, f16_gguf_path: str, quant_type: str = "Q5_K_M") -> str:
        base_name = os.path.splitext(f16_gguf_path)[0]
        output_path = f"{base_name}_{quant_type}.gguf"
        return output_path

    @staticmethod
    def get_metadata(
        d_model: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        vocab_size: int = 32000,
        rope_theta: float = 1_000_000.0,
        max_len: int = 32768,
        intermediate_size: int = 1536,
    ) -> dict:
        return {
            "general.architecture": "tri_transformer_o_branch",
            "tri_transformer.d_model": d_model,
            "tri_transformer.num_heads": num_heads,
            "tri_transformer.num_kv_heads": num_kv_heads,
            "tri_transformer.vocab_size": vocab_size,
            "tri_transformer.rope_theta": rope_theta,
            "tri_transformer.context_length": max_len,
            "tri_transformer.intermediate_size": intermediate_size,
        }
