from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# https://github.com/pytorch/pytorch/pull/110493
class CosineAnnealingWarmRestartsFixedProbably(CosineAnnealingWarmRestarts):
  _resume_handled: bool = False
  def step(self, epoch=None):
    # CosineAnnealingWarmRestarts has a bug with resuming from anything except small values of last_epoch,
    # because it employs an `if` rather than a `while` here:
    # https://github.com/pytorch/pytorch/blob/efb73fe8e4413a0d6db078e85c7ed7c91f05ca5d/torch/optim/lr_scheduler.py#L1411-L1413
    # we can run the while loop here for it.
    if epoch is None and self.last_epoch >= 0 and not self._resume_handled:
      self.T_cur = self.T_cur + 1
      while self.T_cur >= self.T_i:
        self.T_cur = self.T_cur - self.T_i
        self.T_i = self.T_i * self.T_mult
      self.T_cur = self.T_cur - 1
      self._resume_handled = True
    super().step(epoch)
    