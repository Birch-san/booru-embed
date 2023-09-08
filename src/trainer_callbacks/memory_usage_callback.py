# https://pypi.org/project/nvidia-ml-py/
# pip install nvidia-ml-py
import pynvml as nvml
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import NamedTuple
import torch
from logging import getLogger

logger = getLogger(__name__)

def to_MiB(bytes: int) -> int:
  return bytes >> 20

class NVMLMemoryStats(NamedTuple):
  used_bytes: int
  total_bytes: int

def nvml_memory_usage(handle) -> NVMLMemoryStats:
  fb_info = nvml.nvmlDeviceGetMemoryInfo(handle)
  return NVMLMemoryStats(used_bytes=fb_info.used, total_bytes=fb_info.total)

class TorchMemoryStats(NamedTuple):
  used_bytes: int
  used_plus_reserved_bytes: int

def torch_memory_usage(device=0) -> TorchMemoryStats:
  used = torch.cuda.memory_allocated(device)
  used_plus_reserved = torch.cuda.memory_reserved(device)
  return TorchMemoryStats(used_bytes=used, used_plus_reserved_bytes=used_plus_reserved)

class MemoryUsageCallback(TrainerCallback):
  def __init__(self) -> None:
    super().__init__()
    nvml.nvmlInit()
    device_count: int = nvml.nvmlDeviceGetCount()
    self.handles = [nvml.nvmlDeviceGetHandleByIndex(did) for did in range(device_count)]

  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    overall_nvml_used = 0
    overall_nvml_total = 0
    logger.info('NVML memory stats (used+reserved, all processes):')
    for did, handle in enumerate(self.handles):
      used_bytes, total_bytes = nvml_memory_usage(handle)
      overall_nvml_used += used_bytes
      overall_nvml_total += total_bytes
      # you'll notice this is slightly higher than the summary you get in nvidia-smi.
      # that's because it's used + reserved.
      # you can compute used+reserved via nvidia-smi yourself:
      # nvidia-smi -i 0 -q -d MEMORY
      logger.info(f'  Device {did}: Used {to_MiB(used_bytes)}MiB / {to_MiB(total_bytes)}MiB')
    if len(self.handles) > 1:
      logger.info(f'  Overall: Used {to_MiB(overall_nvml_used)}MiB / {to_MiB(overall_nvml_total)}MiB')

    overall_torch_used = 0
    overall_torch_used_plus_reserved_bytes = 0
    logger.info('Torch memory stats (allocated, reserved):')
    for did in range(len(self.handles)):
      used_bytes, used_plus_reserved_bytes = torch_memory_usage(did)
      overall_torch_used += used_bytes
      overall_torch_used_plus_reserved_bytes += used_plus_reserved_bytes
      # Allocated/resident includes stuff like optimizer state
      # Reserved includes temporary state like gradients
      logger.info(f'  Device {did}: Used {to_MiB(used_plus_reserved_bytes)}MiB (Allocated: {to_MiB(used_bytes)}MiB, Reserved {to_MiB(used_plus_reserved_bytes-used_bytes)}MiB)')
    if len(self.handles) > 1:
      logger.info(f'  Overall: Used {to_MiB(overall_torch_used_plus_reserved_bytes)}MiB (Allocated: {to_MiB(overall_torch_used)}MiB, Reserved {to_MiB(overall_torch_used_plus_reserved_bytes-overall_torch_used)}MiB)')

