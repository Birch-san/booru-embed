import torch
from torch import ones
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from ..model.modeling_t5_booru import T5BooruForMaskedLM
from ..model.sigma_reparam import SReparam
from ..booru_collator import BooruBatchData
from contextlib import nullcontext
from logging import getLogger
from dataclasses import dataclass, field

logger = getLogger(__name__)

@dataclass
class SReparamCallback(TrainerCallback):
  amp_context: torch.cuda.amp.autocast|nullcontext = field(default_factory=nullcontext)

  def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    model: T5BooruForMaskedLM = kwargs['model']
    # batch_shape = [args.per_device_train_batch_size, model.config.max_ctx_len]
    batch_shape = [1, 8]
    device=model.device
    batch = BooruBatchData(
      input_ids=ones(*batch_shape, dtype=torch.int16, device=device),
      attention_mask=ones(*batch_shape, dtype=torch.bool, device=device),
      labels=ones(*batch_shape, dtype=torch.int16, device=device),
      decoder_input_ids=ones(*batch_shape, dtype=torch.int16, device=device),
      decoder_attention_mask=ones(*batch_shape, dtype=torch.bool, device=device),
    )
    with self.amp_context:
      SReparam.init_all_(model, **batch)
  
  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    model: T5BooruForMaskedLM = kwargs['model']
    with self.amp_context:
      SReparam.update_all_(model)