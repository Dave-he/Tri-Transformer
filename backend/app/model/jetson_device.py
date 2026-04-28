import os
import subprocess
from dataclasses import dataclass


@dataclass
class JetsonDeviceInfo:
    is_jetson: bool
    gpu_name: str
    cuda_version: str
    total_memory_gb: float
    is_shared_memory: bool
    cuda_cores: int


JETSON_TEGRA_RELEASE = "/etc/nv_tegra_release"


def detect_jetson_device() -> JetsonDeviceInfo:
    is_jetson = os.path.exists(JETSON_TEGRA_RELEASE)
    if not is_jetson:
        return JetsonDeviceInfo(
            is_jetson=False,
            gpu_name="Unknown",
            cuda_version="0",
            total_memory_gb=0,
            is_shared_memory=False,
            cuda_cores=0,
        )

    gpu_name = "NVIDIA Jetson Nano"
    cuda_version = "10.2"
    total_memory_gb, cuda_cores = _get_memory_info()

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            gpu_name = parts[0].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            cuda_version = result.stdout.strip().split(".")[0] + "." + result.stdout.strip().split(".")[1]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return JetsonDeviceInfo(
        is_jetson=True,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        total_memory_gb=total_memory_gb,
        is_shared_memory=True,
        cuda_cores=cuda_cores,
    )


def _get_memory_info() -> tuple:
    meminfo = _read_proc_meminfo()
    mem_total_kb = meminfo.get("MemTotal", 8192000)
    total_memory_gb = round(mem_total_kb / 1024 / 1024, 1)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=num_cuda_cores", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            cuda_cores = int(result.stdout.strip())
        else:
            cuda_cores = 128
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        cuda_cores = 128

    return total_memory_gb, cuda_cores


def _read_proc_meminfo() -> dict:
    meminfo = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        meminfo[key] = int(parts[1])
                    except ValueError:
                        pass
    except FileNotFoundError:
        meminfo = {"MemTotal": 8192000, "MemAvailable": 4096000}
    return meminfo


def get_memory_usage_pct() -> float:
    meminfo = _read_proc_meminfo()
    mem_total = meminfo.get("MemTotal", 1)
    mem_available = meminfo.get("MemAvailable", 0)
    if mem_total == 0:
        return 0.0
    used_pct = ((mem_total - mem_available) / mem_total) * 100
    return round(used_pct, 1)
