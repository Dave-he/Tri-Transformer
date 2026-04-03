import json
import warnings


class TrainingLogger:
    def __init__(self, log_file: str = None):
        self._history = []
        self._log_file = log_file

    def log(self, metrics: dict) -> None:
        self._history.append(dict(metrics))
        if self._log_file is not None:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics) + "\n")
            except Exception as e:
                warnings.warn(f"TrainingLogger write failed: {e}")

    def get_history(self) -> list:
        return list(self._history)
