import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

from app.model.gguf_converter import GGUFConverter, extract_o_branch_weights


class DummyDecoder(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(self.embed(x))


class TestExtractOBranthWeights(unittest.TestCase):
    def test_extracts_streaming_decoder_state_dict(self):
        decoder = DummyDecoder(vocab_size=32000, d_model=256)
        model_state = {"o_branch.streaming_decoder.embed.weight": decoder.embed.weight,
                       "o_branch.streaming_decoder.linear.weight": decoder.linear.weight,
                       "i_branch.some.weight": torch.zeros(1)}
        extracted = extract_o_branch_weights(model_state, prefix="o_branch.streaming_decoder.")
        self.assertIn("embed.weight", extracted)
        self.assertNotIn("i_branch.some.weight", extracted)

    def test_empty_when_no_matching_prefix(self):
        model_state = {"i_branch.some.weight": torch.zeros(1)}
        extracted = extract_o_branch_weights(model_state, prefix="o_branch.streaming_decoder.")
        self.assertEqual(len(extracted), 0)


class TestGGUFConverter(unittest.TestCase):
    def test_init_with_checkpoint_path(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": {"test": torch.zeros(1)}, "epoch": 0, "loss": 1.0}, f.name)
            converter = GGUFConverter(checkpoint_path=f.name)
            self.assertIsNotNone(converter.checkpoint_path)
            os.unlink(f.name)

    def test_extract_weights_returns_dict(self):
        decoder = DummyDecoder()
        state = {"o_branch.streaming_decoder.embed.weight": decoder.embed.weight.data,
                 "o_branch.streaming_decoder.linear.weight": decoder.linear.weight.data}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": state, "epoch": 0, "loss": 1.0}, f.name)
            converter = GGUFConverter(checkpoint_path=f.name)
            weights = converter.extract_o_branch_weights()
            self.assertIsInstance(weights, dict)
            self.assertGreater(len(weights), 0)
            os.unlink(f.name)

    @patch("app.model.gguf_converter.GGUFConverter._run_hf_to_gguf")
    @patch("app.model.gguf_converter.GGUFConverter._save_safetensors")
    def test_convert_calls_steps(self, mock_save, mock_run):
        mock_save.return_value = "/tmp/dummy_dir"
        mock_run.return_value = "/tmp/dummy.gguf"
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": {}, "epoch": 0, "loss": 1.0}, f.name)
            converter = GGUFConverter(checkpoint_path=f.name)
            result = converter.convert(output_dir="/tmp", model_name="test_model")
            mock_save.assert_called_once()
            os.unlink(f.name)

    def test_get_metadata_includes_tri_transformer_fields(self):
        metadata = GGUFConverter.get_metadata(
            d_model=512, num_heads=8, num_kv_heads=2, vocab_size=32000,
            rope_theta=1_000_000, max_len=32768,
        )
        self.assertEqual(metadata["tri_transformer.d_model"], 512)
        self.assertEqual(metadata["tri_transformer.num_heads"], 8)
        self.assertEqual(metadata["general.architecture"], "tri_transformer_o_branch")


if __name__ == "__main__":
    unittest.main()
