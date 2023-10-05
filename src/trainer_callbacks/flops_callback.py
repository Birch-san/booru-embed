from torch.utils.flop_counter import FlopCounterMode
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional, TypedDict, Dict
from logging import getLogger
from torch._ops import OpOverloadPacket

from ..flop_mapping import FlopCustomMapping
from ..welford import Welford

logger = getLogger(__name__)

FlopMetrics = TypedDict("FlopMetrics", {
  'perf/flops': float,
  'perf/flops_avg': float,
})

class FlopsCallback(TrainerCallback):
  flop_counter: FlopCounterMode
  step_tic: Optional[float] = None
  flops_avg: Welford
  unflushed_flos = 0
  unflushed_secs = 0.
  last_log_step = 0.

  metrics: FlopMetrics = {
    'perf/flops': 0.,
    'perf/flops_avg': 0.,
  }

  def __init__(self, add_xformers_mappings=False) -> None:
    super().__init__()
    custom_mapping: Dict[OpOverloadPacket, FlopCustomMapping] = {}
    if add_xformers_mappings:
      from xformers.ops import MemoryEfficientAttentionCutlassOp
      from ..xformers_flop_mappings import cutlass_fwd_flop, cutlass_bwd_flop

      # xformers has more memory efficient attention backends than just cutlassF, but that's the only one I needed (supports attention bias)
      cutlass_fwd, cutlass_bwd = MemoryEfficientAttentionCutlassOp

      custom_mapping.update({
        cutlass_fwd.OPERATOR: cutlass_fwd_flop,
        cutlass_bwd.OPERATOR: cutlass_bwd_flop,
      })
    self.flops_avg = Welford()
    self.flop_counter = FlopCounterMode(display=False, custom_mapping=custom_mapping)

  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.step_tic = perf_counter()
    self.flop_counter.__enter__()
  
  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    step_duration_secs: float = perf_counter() - self.step_tic
    self.flop_counter.__exit__(None, None, None)

    flos_this_step: int = sum(self.flop_counter.get_flop_counts()['Global'].values())
    flops_this_step: float = flos_this_step / step_duration_secs

    self.flops_avg.update(flops_this_step)

    self.unflushed_flos += flos_this_step
    self.unflushed_secs += step_duration_secs
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    flops: float = self.unflushed_flos / self.unflushed_secs
    flops_avg: float = self.flops_avg.mean

    self.metrics['perf/flops'] = flops
    self.metrics['perf/flops_avg'] = flops_avg

    steps_since_last_log: int = state.global_step - self.last_log_step

    logger.info(f'step %s TFLOPs: %.02f secs: %.02f (%d-step avg) / TFLOPs: %.02f (overall avg)', state.global_step, flops/1000**4, self.unflushed_secs/steps_since_last_log, steps_since_last_log, flops_avg/1000**4)
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.metrics, step=state.global_step, commit=False)

    self.last_log_step = state.global_step
    self.unflushed_secs = self.unflushed_flos = 0