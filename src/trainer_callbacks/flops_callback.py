from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional, TypedDict
from logging import getLogger

logger = getLogger(__name__)

FlopMetrics = TypedDict("FlopMetrics", {
  'perf/flos': float,
  'perf/total_flos': float,
  'perf/flops': float,
})

class FlopsCallback(TrainerCallback):
  train_begin: Optional[float] = None
  prev_total_flos = 0
  last_log_tic: Optional[float] = 0
  metrics: FlopMetrics = {
    'perf/flos': 0.,
    # RECOMMENDED: ${perf/total_flos} / ${perf/train_duration}
    'perf/total_flos': 0.,
    'perf/flops': 0.,
  }
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    now: float = perf_counter()
    secs_since_last_log: float = now - self.last_log_tic
    flos_since_last_log: float = state.total_flos - self.prev_total_flos
    self.prev_total_flos = state.total_flos

    self.metrics['perf/flos'] = flos_since_last_log
    self.metrics['perf/total_flos'] = state.total_flos
    self.metrics['perf/flops'] = flos_since_last_log / secs_since_last_log

    logger.info(f'step %s TFLOPs: %.02f', state.global_step, self.metrics['perf/flops']/1000**4)
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.metrics, step=state.global_step, commit=False)