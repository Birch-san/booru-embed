from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional, TypedDict
from logging import getLogger

logger = getLogger(__name__)

TrainDurationMetrics = TypedDict("TrainDurationMetrics", {
    # ensures that even step0 includes the cost of iterating epoch_iterator. probably the fairest.
    'perf/train_duration': float,
    # avoids measuring skip_first_batches(). probably desirable if resuming.
    'perf/since_step0_duration': float,
    # focuses on measuring time spent on model and optimizer
    'perf/intrastep_duration': float,
    # focuses on measuring time spent iterating dataloader
    'perf/interstep_duration': float,
    # divide by (global_step+1) to get average intrastep duration
    'perf/all_intrastep_duration': float,
    # divide by global_step to get average interstep duration
    'perf/all_interstep_duration': float,
})

class TrainDurationCallback(TrainerCallback):
  """
  Measures the time that has passed since on_train_begin.
  This can help you exclude the walltime spent on initialization.
  """
  train_begin: Optional[float] = None
  step0_begin: Optional[float] = None
  step_begin: Optional[float] = None
  step_end: Optional[float] = None
  metrics: TrainDurationMetrics = {
    'perf/train_duration': 0.,
    'perf/since_step0_duration': 0.,
    'perf/intrastep_duration': 0.,
    'perf/interstep_duration': 0.,
    'perf/all_intrastep_duration': 0.,
    'perf/all_interstep_duration': 0.,
  }
  
  def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.train_begin = perf_counter()
  
  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.step_begin = perf_counter()
    if self.step0_begin is None:
      self.step0_begin = self.step_begin
    if self.step_end is not None:
      self.metrics['perf/interstep_duration'] = self.step_begin - self.step_end
      self.metrics['perf/all_interstep_duration'] += self.metrics['perf/interstep_duration']
  
  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.step_end = perf_counter()
    self.metrics['perf/intrastep_duration'] = self.step_end - self.step_begin
    self.metrics['perf/all_intrastep_duration'] += self.metrics['perf/intrastep_duration']
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    now: float = perf_counter()
    self.metrics['perf/train_duration'] = now - self.train_begin
    self.metrics['perf/since_step0_duration'] = now - self.step0_begin
    logger.info(f'step %d train duration secs: %.02f', state.global_step, self.metrics['perf/train_duration'])
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.metrics, step=state.global_step, commit=False)