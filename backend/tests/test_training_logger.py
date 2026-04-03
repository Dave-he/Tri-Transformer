import json
import os
import warnings

import pytest

from app.services.train.training_logger import TrainingLogger


class TestTrainingLoggerBasic:
    def test_log_stores_in_history(self, tmp_path):
        logger = TrainingLogger()
        logger.log({"epoch": 1, "loss": 0.5, "lr": 1e-4})
        logger.log({"epoch": 2, "loss": 0.4, "lr": 1e-4})
        history = logger.get_history()
        assert len(history) == 2

    def test_log_writes_jsonl_file(self, tmp_path):
        log_file = str(tmp_path / "training_log.jsonl")
        logger = TrainingLogger(log_file=log_file)
        logger.log({"epoch": 1, "loss": 0.5, "lr": 1e-4})
        logger.log({"epoch": 2, "loss": 0.4, "lr": 1e-4})
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "epoch" in data
            assert "loss" in data
            assert "lr" in data

    def test_get_history_returns_all_entries(self, tmp_path):
        log_file = str(tmp_path / "log.jsonl")
        logger = TrainingLogger(log_file=log_file)
        for i in range(5):
            logger.log({"epoch": i + 1, "loss": 1.0 / (i + 1), "lr": 1e-4})
        assert len(logger.get_history()) == 5

    def test_log_file_none_memory_mode(self):
        logger = TrainingLogger(log_file=None)
        logger.log({"epoch": 1, "loss": 0.5, "lr": 1e-4})
        assert len(logger.get_history()) == 1


class TestTrainingLoggerErrorHandling:
    def test_write_failure_does_not_raise(self):
        logger = TrainingLogger(log_file="/nonexistent/dir/log.jsonl")
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                logger.log({"epoch": 1, "loss": 0.5, "lr": 1e-4})
            assert any(issubclass(warning.category, UserWarning) for warning in w)
        except Exception as e:
            pytest.fail(f"log() raised an exception: {e}")

    def test_write_failure_still_stores_in_history(self):
        logger = TrainingLogger(log_file="/nonexistent/dir/log.jsonl")
        logger.log({"epoch": 1, "loss": 0.5, "lr": 1e-4})
        assert len(logger.get_history()) == 1
