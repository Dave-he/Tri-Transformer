from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_fake_conversation_dataset(n=10):
    return [{"conversation": [{"role": "user", "content": f"hi {i}"}, {"role": "assistant", "content": f"hello {i}"}]} for i in range(n)]


def _make_fake_belle_dataset(n=10):
    return [{"instruction": f"instruction {i}", "output": f"output {i}"} for i in range(n)]


class TestModelScopeDatasetLoader:
    @patch("app.services.train.dataset_loader.MsDataset")
    def test_load_lccc_returns_dataset(self, MockMsDataset):
        MockMsDataset.load.return_value = _make_fake_conversation_dataset()
        from app.services.train.dataset_loader import ModelScopeDatasetLoader
        loader = ModelScopeDatasetLoader()
        dataset = loader.load("lccc")
        assert dataset is not None
        assert len(dataset) > 0

    @patch("app.services.train.dataset_loader.MsDataset")
    def test_load_belle_returns_dataset(self, MockMsDataset):
        MockMsDataset.load.return_value = _make_fake_belle_dataset()
        from app.services.train.dataset_loader import ModelScopeDatasetLoader
        loader = ModelScopeDatasetLoader()
        dataset = loader.load("belle")
        assert dataset is not None

    @patch("app.services.train.dataset_loader.MsDataset")
    def test_get_dataloader_outputs_tensors(self, MockMsDataset):
        MockMsDataset.load.return_value = _make_fake_conversation_dataset(20)
        from app.services.train.dataset_loader import ModelScopeDatasetLoader
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 256
        mock_tokenizer.encode.side_effect = lambda text, **kw: [ord(c) % 255 + 1 for c in text[:8]]

        loader = ModelScopeDatasetLoader()
        dataset = loader.load("lccc")
        dl = loader.get_dataloader(dataset, tokenizer=mock_tokenizer, batch_size=4, max_len=16)

        batch = next(iter(dl))
        assert len(batch) == 3
        src, tgt_in, tgt_out = batch
        assert src.dtype == torch.long
        assert tgt_in.dtype == torch.long
        assert tgt_out.dtype == torch.long
        assert src.shape[0] == 4

    @patch("app.services.train.dataset_loader.MsDataset")
    def test_tensor_shapes_correct(self, MockMsDataset):
        MockMsDataset.load.return_value = _make_fake_conversation_dataset(20)
        from app.services.train.dataset_loader import ModelScopeDatasetLoader
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 256
        mock_tokenizer.encode.side_effect = lambda text, **kw: [1, 2, 3, 4, 5]

        loader = ModelScopeDatasetLoader()
        dataset = loader.load("lccc")
        dl = loader.get_dataloader(dataset, tokenizer=mock_tokenizer, batch_size=4, max_len=16)

        src, tgt_in, tgt_out = next(iter(dl))
        assert src.shape[1] == 16
        assert tgt_in.shape[1] == 16
        assert tgt_out.shape[1] == 16

    def test_unsupported_dataset_raises(self):
        from app.services.train.dataset_loader import ModelScopeDatasetLoader
        loader = ModelScopeDatasetLoader()
        with pytest.raises(ValueError):
            loader.load("unsupported_dataset_xyz")
