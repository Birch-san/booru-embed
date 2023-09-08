import torch
from typing import NamedTuple

class TorchMemoryStats(NamedTuple):
  used_bytes: int
  used_plus_reserved_bytes: int

def torch_cuda_memory_usage(device=0) -> TorchMemoryStats:
  used = torch.cuda.memory_allocated(device)
  used_plus_reserved = torch.cuda.memory_reserved(device)
  return TorchMemoryStats(used_bytes=used, used_plus_reserved_bytes=used_plus_reserved)