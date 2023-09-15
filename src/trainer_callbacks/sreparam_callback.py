import torch
from torch import ones
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from ..model.modeling_t5_booru import T5BooruForMaskedLM
from ..model.sigma_reparam import SReparam
from ..booru_collator import BooruBatchData
from typing import Optional
from contextlib import nullcontext
from logging import getLogger

logger = getLogger(__name__)

class SReparamCallback(TrainerCallback):
  amp_context: Optional[torch.cuda.amp.autocast|nullcontext] = None

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
    SReparam.init_all_(model, **batch)
  
  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    model: T5BooruForMaskedLM = kwargs['model']
    if self.amp_context is None:
      match args.half_precision_backend:
        case 'auto' | 'cuda':
          amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
          self.amp_context = torch.cuda.amp.autocast(cache_enabled=True, dtype=amp_dtype)
        case _:
          logger.warning(f'half_precision_backend={args.half_precision_backend} not implemented; disabling amp')
          self.amp_context = nullcontext()

    with self.amp_context:
      SReparam.update_all_(model)