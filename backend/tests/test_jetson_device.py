import os
import unittest
from unittest.mock import MagicMock, patch, mock_open

from app.model.jetson_device import JetsonDeviceInfo, detect_jetson_device, get_memory_usage_pct


class TestJetsonDeviceInfo(unittest.TestCase):
    def test_dataclass_fields(self):
        info = JetsonDeviceInfo(
            is_jetson=True,
            gpu_name="NVIDIA Jetson Nano",
            cuda_version="10.2",
            total_memory_gb=8.0,
            is_shared_memory=True,
            cuda_cores=128,
        )
        self.assertTrue(info.is_jetson)
        self.assertEqual(info.gpu_name, "NVIDIA Jetson Nano")
        self.assertEqual(info.cuda_version, "10.2")
        self.assertEqual(info.total_memory_gb, 8.0)
        self.assertTrue(info.is_shared_memory)
        self.assertEqual(info.cuda_cores, 128)

    def test_non_jetson_default(self):
        info = JetsonDeviceInfo(
            is_jetson=False,
            gpu_name="Unknown",
            cuda_version="0",
            total_memory_gb=0,
            is_shared_memory=False,
            cuda_cores=0,
        )
        self.assertFalse(info.is_jetson)


class TestDetectJetsonDevice(unittest.TestCase):
    @patch("os.path.exists")
    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open, read_data="R32 (release), REVISION: 4.3")
    def test_detects_jetson_nano(self, mock_file, mock_subprocess, mock_exists):
        mock_exists.side_effect = lambda p: p == "/etc/nv_tegra_release"
        mock_subprocess.return_value = MagicMock(
            stdout="GPU 0: NVIDIA Jetson Nano (Maxwell) | CUDA 10.2"
        )
        with patch("app.model.jetson_device._get_memory_info", return_value=(8.0, 128)):
            info = detect_jetson_device()
        self.assertTrue(info.is_jetson)
        self.assertEqual(info.cuda_version, "10.2")

    @patch("os.path.exists", return_value=False)
    def test_detects_non_jetson(self, mock_exists):
        info = detect_jetson_device()
        self.assertFalse(info.is_jetson)


class TestGetMemoryUsagePct(unittest.TestCase):
    @patch("app.model.jetson_device._read_proc_meminfo", return_value={"MemTotal": 8192000, "MemAvailable": 2048000})
    def test_returns_percentage(self, mock_meminfo):
        pct = get_memory_usage_pct()
        self.assertIsInstance(pct, float)
        self.assertGreaterEqual(pct, 0)
        self.assertLessEqual(pct, 100)

    @patch("app.model.jetson_device._read_proc_meminfo", return_value={"MemTotal": 8192000, "MemAvailable": 8192000})
    def test_zero_usage_when_all_available(self, mock_meminfo):
        pct = get_memory_usage_pct()
        self.assertAlmostEqual(pct, 0.0, places=1)

    @patch("app.model.jetson_device._read_proc_meminfo", return_value={"MemTotal": 8192000, "MemAvailable": 0})
    def test_full_usage_when_none_available(self, mock_meminfo):
        pct = get_memory_usage_pct()
        self.assertAlmostEqual(pct, 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
