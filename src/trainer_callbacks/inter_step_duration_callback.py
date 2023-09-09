from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional, TypedDict
from logging import getLogger

logger = getLogger(__name__)

InterStepMetrics = TypedDict("InterStepMetrics", {
  'perf/interstep_s': float,
  'perf/interstep_s_avg': float,
})

class InterStepDurationCallback(TrainerCallback):
  """
  Measures how long each on_step_end->on_step_begin takes,
  to tell you how long it takes to iterate to the next batch from your data collator
  """
  interstep_tic: Optional[float] = None
  metrics: InterStepMetrics = {
    'perf/interstep_s': 0.,
    'perf/interstep_s_avg': 0.,
  }
  steps_considered = 0

  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    assert self.interstep_tic is not None
    
    self.metrics['perf/interstep_s'] = perf_counter()-self.interstep_tic
    self.metrics['perf/interstep_s_avg'] = (self.metrics['perf/interstep_s_avg'] * self.steps_considered + self.metrics['perf/interstep_s']) / (self.steps_considered+1)
    self.steps_considered += 1
    self.interstep_tic = None

  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.interstep_tic = perf_counter()
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    logger.info(f'step %s interstep secs: %.02f (avg %.02f)', state.global_step, self.metrics['perf/interstep_s'], self.metrics['perf/interstep_s_avg'])
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log(self.metrics, step=state.global_step, commit=False)