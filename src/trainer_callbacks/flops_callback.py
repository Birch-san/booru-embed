from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional, TypedDict
from logging import getLogger

logger = getLogger(__name__)

FlopMetrics = TypedDict("FlopMetrics", {
  'perf/current_flos': float,
  'perf/total_flos': float,
  'perf/flops': float,
  'perf/flops_avg': float,
})

class FlopsCallback(TrainerCallback):
  log_every_n_steps: int
  train_begin: Optional[float] = None
  prev_total_flos = 0
  last_log_tic: Optional[float] = 0
  metrics: FlopMetrics = {
    'perf/current_flos': 0.,
    'perf/total_flos': 0.,
    'perf/flops': 0.,
    'perf/flops_avg': 0.,
  }

  def __init__(self, log_every_n_steps=0) -> None:
    super().__init__()
    self.log_every_n_steps = log_every_n_steps
  
  def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.train_begin = self.last_log_tic = perf_counter()

  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if state.global_step % self.log_every_n_steps == 0:
      return TrainerControl(should_log=True)
    return control
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    assert self.train_begin is not None
    now: float = perf_counter()
    secs_since_training_began: float = now - self.train_begin
    secs_since_last_log: float = now - self.last_log_tic
    flos_since_last_log: float = state.total_flos - self.prev_total_flos
    self.prev_total_flos = state.total_flos

    self.metrics['perf/current_flos'] = flos_since_last_log
    self.metrics['perf/total_flos'] = state.total_flos
    self.metrics['perf/flops'] = flos_since_last_log / secs_since_last_log
    self.metrics['perf/flops_avg'] = state.total_flos / secs_since_training_began

    logger.info(f'step %s TFLOPs: %.02f (avg %.02f)', state.global_step, self.metrics['perf/flops']/1000**4, self.metrics['perf/flops_avg']/1000**4)
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.metrics, step=state.global_step, commit=False)