from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional, TypedDict
from logging import getLogger

logger = getLogger(__name__)

TrainDurationMetrics = TypedDict("TrainDurationMetrics", {
  'perf/train_duration': float,
})

class TrainDurationCallback(TrainerCallback):
  """
  Measures the time that has passed since on_train_begin.
  This can help you exclude the walltime spent on initialization.
  """
  train_begin: Optional[float] = None
  metrics: TrainDurationMetrics = {
    'perf/train_duration': 0.,
  }
  
  def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.train_begin = perf_counter()
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.metrics['perf/train_duration'] = perf_counter() - self.train_begin
    logger.info(f'step %d run duration secs: %.02f', state.global_step, self.metrics['perf/train_duration'])
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.metrics, step=state.global_step, commit=False)