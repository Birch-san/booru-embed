from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch
from logging import getLogger
from dataclasses import dataclass, field
from typing import Dict

from ..nvml_service import NvmlService
from ..torch_memory import torch_cuda_memory_usage

logger = getLogger(__name__)

def to_MiB(bytes: int) -> int:
  return bytes >> 20

@dataclass
class MemoryUsageCallback(TrainerCallback):
  nvml_service: NvmlService
  torch_cuda_device_count: int = field(init=False)
  log_update: Dict[str, int] = field(default_factory=lambda:{}, init=False)

  def __post_init__(self):
    self.torch_cuda_device_count = torch.cuda.device_count()

  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    overall_nvml_used = 0
    overall_nvml_total = 0
    logger.info('NVML memory stats (used+reserved, all processes):')
    for did in range(self.nvml_service.device_count):
      used_bytes, total_bytes = self.nvml_service.memory_usage(did)
      overall_nvml_used += used_bytes
      overall_nvml_total += total_bytes
      # you'll notice this is slightly higher than the summary you get in nvidia-smi.
      # that's because it's used + reserved.
      # you can compute used+reserved via nvidia-smi yourself:
      # nvidia-smi -i 0 -q -d MEMORY
      logger.info(f'  Device {did}: Used {to_MiB(used_bytes)}MiB / {to_MiB(total_bytes)}MiB')
      self.log_update[f'sys/nvml_mem_used_{did}'] = used_bytes
    if self.nvml_service.device_count > 1:
      logger.info(f'  Overall: Used {to_MiB(overall_nvml_used)}MiB / {to_MiB(overall_nvml_total)}MiB')
      self.log_update['sys/nvml_mem_used_overall'] = used_bytes

    overall_torch_used = 0
    overall_torch_used_plus_reserved_bytes = 0
    logger.info('Torch memory stats (allocated, reserved):')
    for did in range(self.torch_cuda_device_count):
      used_bytes, used_plus_reserved_bytes = torch_cuda_memory_usage(did)
      overall_torch_used += used_bytes
      overall_torch_used_plus_reserved_bytes += used_plus_reserved_bytes
      # Allocated/resident includes stuff like optimizer state
      # Reserved includes temporary state like gradients
      logger.info(f'  Device {did}: Used {to_MiB(used_plus_reserved_bytes)}MiB (Allocated: {to_MiB(used_bytes)}MiB, Reserved {to_MiB(used_plus_reserved_bytes-used_bytes)}MiB)')
      self.log_update[f'sys/torch_mem_used_{did}'] = used_bytes
      self.log_update[f'sys/torch_mem_used_plus_reserved_{did}'] = used_plus_reserved_bytes
    if self.nvml_service.device_count > 1:
      logger.info(f'  Overall: Used {to_MiB(overall_torch_used_plus_reserved_bytes)}MiB (Allocated: {to_MiB(overall_torch_used)}MiB, Reserved {to_MiB(overall_torch_used_plus_reserved_bytes-overall_torch_used)}MiB)')
      self.log_update['sys/torch_mem_used_total'] = overall_torch_used
      self.log_update['sys/torch_mem_used_plus_reserved_total'] = overall_torch_used_plus_reserved_bytes
    
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.log_update, step=state.global_step, commit=False)

