from torch.optim import Optimizer
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from .schedule.cosine_annealing_warm_restarts_decay_warmup import CosineAnnealingWarmRestartsDecayWarmup

def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    lr: float = max(current_step, num_warmup_steps)**-.5
    return lr


def get_inverse_sqrt_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = num_warmup_steps

    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_cosine_annealing_warm_restarts_decay_with_warmup_schedule(
    optimizer: Optimizer, num_warmup_steps: int, T_0: int, eta_min: float, last_epoch: int = -1,
    decay: float = .95, warmup_start_factor: float = 0.,
    T_mult: int = 2,
) -> CosineAnnealingWarmRestartsDecayWarmup:
    return CosineAnnealingWarmRestartsDecayWarmup(
        optimizer=optimizer,
        eta_min=eta_min,
        warmup=num_warmup_steps,
        warmup_start_factor=warmup_start_factor,
        T_0=T_0,
        T_mult=T_mult,
        decay=decay,
        last_epoch=last_epoch,
    )

def get_cosine_annealing_with_warmup_schedule(
    optimizer: Optimizer, num_warmup_steps: int, total_steps: int, eta_min: float, last_epoch: int = -1,
    warmup_start_factor: float = 0.,
) -> CosineAnnealingWarmRestartsDecayWarmup:
    # whilst it would be simpler to express this as SequentialLR(LinearLR(), CosineAnnealingLR()),
    # I found that those do not tolerate resuming training correctly.
    # so we express a very similar schedule using my far-more-complicated but resumption-tested schedule
    return CosineAnnealingWarmRestartsDecayWarmup(
        optimizer=optimizer,
        warmup=num_warmup_steps,
        warmup_start_factor=warmup_start_factor,
        eta_min=eta_min,
        T_0=total_steps-num_warmup_steps+1,
        T_mult=1,
        decay=.001,
        last_epoch=last_epoch,
    )
