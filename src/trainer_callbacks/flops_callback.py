from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from time import perf_counter
from typing import Optional
from logging import getLogger

logger = getLogger(__name__)

class FlopsCallback(TrainerCallback):
  log_every_n_steps: int
  step_tic: Optional[float] = None
  step_duration: Optional[float] = None
  all_steps_secs = 0
  latest_flos = 0

  def __init__(self, log_every_n_steps=0) -> None:
    super().__init__()
    self.log_every_n_steps = log_every_n_steps

  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    self.step_tic = perf_counter()

  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    assert self.step_tic is not None
    self.step_duration: float = perf_counter()-self.step_tic
    self.step_tic = None
    if state.global_step % self.log_every_n_steps == 0:
      return TrainerControl(should_log=True)
    return control
  
  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    assert self.step_duration is not None
    step_flos: float = state.total_flos - self.latest_flos
    self.latest_flos = state.total_flos

    self.all_steps_secs += self.step_duration

    step_flops: float = step_flos/self.step_duration
    all_steps_flops: float = state.total_flos/self.all_steps_secs

    logger.info(f'step %s TFLOPs: %.02f (avg %.02f)', state.global_step, step_flops/1000**4, all_steps_flops/1000**4)
    self.step_duration = None
    if args.report_to and 'wandb' in args.report_to:
      import wandb
      wandb.log({
        'sys/flops': step_flops,
        'sys/flops_avg': all_steps_flops,
      }, step=state.global_step, commit=False)