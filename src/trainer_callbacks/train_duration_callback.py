from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional, TypedDict
from logging import getLogger

logger = getLogger(__name__)

TrainDurationMetrics = TypedDict("TrainDurationMetrics", {
    'perf/train_duration': float,
    'perf/intrastep_duration': float,
    'perf/interstep_duration': float,
    'perf/all_intrastep_duration': float,
    'perf/all_interstep_duration': float,
})

class TrainDurationCallback(TrainerCallback):
  """
  Measures the time that has passed since on_train_begin.
  This can help you exclude the walltime spent on initialization.
  """
  train_begin: Optional[float] = None
  step_begin: Optional[float] = None
  step_end: Optional[float] = None
  metrics: TrainDurationMetrics = {
    'perf/train_duration': 0.,
    'perf/intrastep_duration': 0.,
    'perf/interstep_duration': 0.,
    'perf/all_intrastep_duration': 0.,
    'perf/all_interstep_duration': 0.,
  }
  
  def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.train_begin = perf_counter()
  
  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.step_begin = perf_counter()
    if self.step_end is not None:
      self.metrics['perf/interstep_duration'] = self.step_begin - self.step_end
      self.metrics['perf/all_interstep_duration'] += self.metrics['perf/interstep_duration']
  
  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.metrics['perf/intrastep_duration'] = perf_counter() - self.step_begin
    self.metrics['perf/all_intrastep_duration'] += self.metrics['perf/intrastep_duration']
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.metrics['perf/train_duration'] = perf_counter() - self.train_begin
    logger.info(f'step %d train duration secs: %.02f', state.global_step, self.metrics['perf/train_duration'])
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.metrics, step=state.global_step, commit=False)