from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional, TypedDict
from logging import getLogger

logger = getLogger(__name__)

IntraStepMetrics = TypedDict("IntraStepMetrics", {
  'perf/intrastep_s': float,
  'perf/intrastep_s_avg': float,
})

class IntraStepDurationCallback(TrainerCallback):
  """
  Measures how long each on_step_begin->on_step_end takes,
  to tell you how long it takes to iterate to the next batch from your data collator
  """
  intrastep_tic: Optional[float] = None
  metrics: IntraStepMetrics = {
    'perf/intrastep_s': 0.,
    'perf/intrastep_s_avg': 0.,
  }
  steps_considered = 0

  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.intrastep_tic = perf_counter()

  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    assert self.intrastep_tic is not None
    
    self.metrics['perf/intrastep_s'] = perf_counter()-self.intrastep_tic
    self.metrics['perf/intrastep_s_avg'] = (self.metrics['perf/intrastep_s_avg'] * self.steps_considered + self.metrics['perf/intrastep_s']) / (self.steps_considered+1)
    self.steps_considered += 1
    self.intrastep_tic = None
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    logger.info(f'step %s intrastep secs: %.02f (avg %.02f)', state.global_step, self.metrics['perf/intrastep_s'], self.metrics['perf/intrastep_s_avg'])
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.metrics, step=state.global_step, commit=False)