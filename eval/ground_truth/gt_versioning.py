from typing import List, Dict, Any, Optional
import json
import hashlib
import os
from datetime import datetime


class GTVersioning:
    def __init__(self, storage_dir: str = "eval/data/gt_versions"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _compute_version_id(self, dataset: List[Dict[str, Any]]) -> str:
        content = json.dumps(dataset, sort_keys=True, ensure_ascii=False)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}_{hash_val}"

    def save(self, dataset: List[Dict[str, Any]]) -> str:
        version_id = self._compute_version_id(dataset)
        file_path = os.path.join(self.storage_dir, f"{version_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"version": version_id, "data": dataset}, f, ensure_ascii=False, indent=2)
        return version_id

    def load(self, version_id: str) -> List[Dict[str, Any]]:
        file_path = os.path.join(self.storage_dir, f"{version_id}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return content["data"]

    def export_jsonlines(self, version_id: str, output_path: str) -> None:
        dataset = self.load(version_id)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def list_versions(self) -> List[str]:
        files = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
        return [f.replace(".json", "") for f in sorted(files)]
