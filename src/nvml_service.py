# https://pypi.org/project/nvidia-ml-py/
# pip install nvidia-ml-py
import pynvml as nvml
from typing import NamedTuple

class NVMLMemoryStats(NamedTuple):
  used_bytes: int
  total_bytes: int

class NvmlService:
  device_count: int
  def __init__(self) -> None:
    nvml.nvmlInit()
    self.device_count = nvml.nvmlDeviceGetCount()
    self._handles = [nvml.nvmlDeviceGetHandleByIndex(did) for did in range(self.device_count)]
  
  def memory_usage(self, device_id: int) -> NVMLMemoryStats:
    handle = self._handles[device_id]
    fb_info = nvml.nvmlDeviceGetMemoryInfo(handle)
    return NVMLMemoryStats(used_bytes=fb_info.used, total_bytes=fb_info.total)