from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from modelscope.msdatasets import MsDataset
    _MODELSCOPE_AVAILABLE = True
except ImportError:
    MsDataset = None
    _MODELSCOPE_AVAILABLE = False

DATASET_REGISTRY = {
    "lccc": {
        "id": "AI-ModelScope/LCCC-base-split",
        "split": "train",
        "text_fields": ["conversation"],
        "format": "conversation",
    },
    "belle": {
        "id": "BelleGroup/train_0.5M_CN",
        "split": "train",
        "text_fields": ["instruction", "output"],
        "format": "instruction",
    },
}


class ConversationDataset(Dataset):
    def __init__(self, records, tokenizer, max_len: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def _pad_or_truncate(self, ids: list, length: int) -> list:
        ids = ids[:length]
        ids = ids + [0] * (length - len(ids))
        return ids

    def __getitem__(self, idx):
        record = self.records[idx]

        if isinstance(record, dict) and "conversation" in record:
            turns = record["conversation"]
            src_text = " ".join(t.get("content", "") for t in turns[:-1]) if len(turns) > 1 else ""
            tgt_text = turns[-1].get("content", "") if turns else ""
        elif isinstance(record, dict) and "instruction" in record:
            src_text = record.get("instruction", "")
            tgt_text = record.get("output", "")
        else:
            src_text = str(record)
            tgt_text = str(record)

        src_ids = self._pad_or_truncate(
            self.tokenizer.encode(src_text, max_length=self.max_len, truncation=True),
            self.max_len,
        )
        tgt_ids = self.tokenizer.encode(tgt_text, max_length=self.max_len + 1, truncation=True)
        tgt_in_ids = self._pad_or_truncate(tgt_ids[:-1] if len(tgt_ids) > 1 else tgt_ids, self.max_len)
        tgt_out_ids = self._pad_or_truncate(tgt_ids[1:] if len(tgt_ids) > 1 else tgt_ids, self.max_len)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_in_ids, dtype=torch.long),
            torch.tensor(tgt_out_ids, dtype=torch.long),
        )


class ModelScopeDatasetLoader:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def load(self, dataset_name: str, split: str = "train", max_samples: int = 50000) -> list:
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {list(DATASET_REGISTRY.keys())}")

        config = DATASET_REGISTRY[dataset_name]

        if not _MODELSCOPE_AVAILABLE:
            return self._generate_dummy_data(dataset_name, n=min(max_samples, 100))

        try:
            ds = MsDataset.load(
                config["id"],
                split=config["split"],
                cache_dir=self.cache_dir,
            )
            records = list(ds)
            return records[:max_samples]
        except Exception:
            return self._generate_dummy_data(dataset_name, n=min(max_samples, 100))

    def _generate_dummy_data(self, dataset_name: str, n: int = 100) -> list:
        config = DATASET_REGISTRY[dataset_name]
        if config["format"] == "conversation":
            return [
                {"conversation": [
                    {"role": "user", "content": f"dummy question {i}"},
                    {"role": "assistant", "content": f"dummy answer {i}"},
                ]}
                for i in range(n)
            ]
        return [
            {"instruction": f"instruction {i}", "output": f"output {i}"}
            for i in range(n)
        ]

    def get_dataloader(
        self,
        dataset: list,
        tokenizer,
        batch_size: int = 8,
        max_len: int = 128,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        conv_dataset = ConversationDataset(dataset, tokenizer=tokenizer, max_len=max_len)
        return DataLoader(
            conv_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
