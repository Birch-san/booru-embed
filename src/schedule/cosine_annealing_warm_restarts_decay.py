import math
from typing import Optional, List
from torch.optim import Optimizer
from .cosine_annealing_warm_restarts_patched_subclass import CosineAnnealingWarmRestartsFixedProbably

# by Noxel, under CC BY-SA 4.0 license
# https://stackoverflow.com/a/73747249/5257399
class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestartsFixedProbably):
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
  decay: float
  initial_lrs: List[float]
  def __init__(
    self,
    optimizer: Optimizer,
    T_0: int,
    T_mult: int = 1,
    eta_min: float = 0.,
    last_epoch: int = -1,
    verbose = False,
    decay: float = 1.,
  ):
    self.decay = decay
    super().__init__(
      optimizer,
      T_0,
      T_mult=T_mult,
      eta_min=eta_min,
      last_epoch=last_epoch,
      verbose=verbose,
    )
    self.initial_lrs = self.base_lrs
  
  def step(self, epoch: Optional[int] = None):
    if epoch == None:
      if self.T_cur + 1 == self.T_i:
        if self.verbose:
          print("multiplying base_lrs by {:.4f}".format(self.decay))
        self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
    else:
      if epoch < 0:
        raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
      if epoch >= self.T_0:
        if self.T_mult == 1:
          n = int(epoch / self.T_0)
        else:
          n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
      else:
        n = 0
      
      self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

    super().step(epoch)