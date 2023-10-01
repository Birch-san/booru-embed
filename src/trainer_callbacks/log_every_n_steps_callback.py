from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from dataclasses import dataclass

@dataclass
class LogEveryNStepsCallback(TrainerCallback):
  log_every_n_steps: int

  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if state.global_step % self.log_every_n_steps == 0:
      return TrainerControl(should_log=True)
    return control