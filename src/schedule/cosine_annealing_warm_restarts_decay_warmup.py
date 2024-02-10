from __future__ import annotations
import math
from typing import Optional, List
import warnings
from torch.optim import Optimizer

from .cosine_annealing_warm_restarts_decay import CosineAnnealingWarmRestartsDecay

class _enable_get_lr_call:
  o: CosineAnnealingWarmRestartsDecayWarmup
  def __init__(self, o: CosineAnnealingWarmRestartsDecayWarmup):
    self.o = o

  def __enter__(self):
    self.o._get_lr_called_within_step = True
    return self
  
  def __exit__(self, type, value, traceback):
    self.o._get_lr_called_within_step = False
    return self

# by Alex Birch
# adds linear warmup to CosineAnnealingWarmRestartsDecay schedule
# note: a simpler design would be to make a delegating scheduler like SequentialLR,
# and compose [LinearLR, CosineAnnealingWarmRestartsDecay] together
class CosineAnnealingWarmRestartsDecayWarmup(CosineAnnealingWarmRestartsDecay):
  r"""Set the learning rate of each parameter group using a cosine annealing
  schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
  is the number of epochs since the last restart and :math:`T_{i}` is the number
  of epochs between two warm restarts in SGDR:

  .. math::
      \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
      \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

  When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
  When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

  It has been proposed in
  `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

  Args:
      optimizer (Optimizer): Wrapped optimizer.
      T_0 (int): Number of iterations for the first restart.
      T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
      eta_min (float, optional): Minimum learning rate. Default: 0.
      last_epoch (int, optional): The index of last epoch. Default: -1.
      verbose (bool): If ``True``, prints a message to stdout for
          each update. Default: ``False``.

  .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
      https://arxiv.org/abs/1608.03983
  """
  warmup: int
  warmup_start_factor: int
  _last_epoch: int
  _get_lr_called_within_step: bool

  def __init__(
    self,
    optimizer: Optimizer,
    T_0: int,
    warmup: int,
    warmup_start_factor: float,
    T_mult: int = 1,
    eta_min: float = 0.,
    last_epoch: int = -1,
    verbose = False,
    decay: float = 1.,
  ):
    self.warmup = warmup
    self.warmup_start_factor = warmup_start_factor
    self._last_epoch = last_epoch
    super().__init__(
      optimizer,
      T_0,
      T_mult=T_mult,
      eta_min=eta_min,
      last_epoch=max(last_epoch-warmup, -1),
      verbose=verbose,
      decay=decay,
    )
    self.initial_lrs = self.base_lrs
  
  def step(self, epoch: Optional[float] = None):
    if epoch == None:
      self._last_epoch = self._last_epoch + 1
    else:
      self._last_epoch = math.floor(epoch)

    if epoch is not None and epoch >= self.warmup or epoch is None and self._last_epoch >= self.warmup:
      super().step(epoch)
    else:
      with _enable_get_lr_call(self):
        for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
          param_group, lr = data
          param_group['lr'] = lr
          self.print_lr(self.verbose, i, lr, epoch)

      self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
  
  def get_lr(self) -> List[float]:
    if not self._get_lr_called_within_step:
      warnings.warn("To get the last learning rate computed by the scheduler, "
                      "please use `get_last_lr()`.", UserWarning)
    
    if self._last_epoch < self.warmup:
      return [base_lr * (self.warmup_start_factor + (1 - self.warmup_start_factor) * self._last_epoch / self.warmup)
              for base_lr in self.base_lrs]

    return super().get_lr()