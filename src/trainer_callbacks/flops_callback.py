from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional
from logging import getLogger

logger = getLogger(__name__)

class FlopsCallback(TrainerCallback):
  step_tic: Optional[float] = None
  step_duration: Optional[float] = None
  all_steps_secs = 0
  latest_flos = 0

  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.step_tic = perf_counter()

  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    assert self.step_tic is not None
    self.step_duration: float = perf_counter()-self.step_tic
    self.step_tic = None
    return TrainerControl(should_log=True)
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    assert self.step_duration is not None
    step_flos: float = state.total_flos - self.latest_flos
    self.latest_flos = state.total_flos

    self.all_steps_secs += self.step_duration

    step_flops: float = step_flos/self.step_duration
    all_steps_flops: float = state.total_flos/self.all_steps_secs

    logger.info(f'step {state.global_step} TFLOPs: {step_flops/1000**4:.02f} (avg {all_steps_flops/1000**4:.02f})')
    self.step_duration = None