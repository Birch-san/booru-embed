from typing import NamedTuple, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class OptimizerAndScheduler(NamedTuple):
  optimizer: Optional[Optimizer]
  schedule: Optional[LambdaLR]