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
    if model.config.s_reparam_config['register_v_during_construction']:
      with self.amp_context:
        SReparam.init_all_statically(model)
    else:
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
        SReparam.init_all_via_trace(model, **batch)
  
  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    model: T5BooruForMaskedLM = kwargs['model']
    with self.amp_context:
      SReparam.update_all_(model)
  
  # debugging; don't commit
  # def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
  #   if args.report_to and 'wandb' in args.report_to:
  #     model: T5BooruForMaskedLM = kwargs['model']
  #     import wandb
  #     wandb.log({
  #       'sreparam/lm_head_sigma': model.lm_head.sigma.item(),
  #       'sreparam/lm_head_weight_mean': model.lm_head.op.weight.mean().item(),
  #       'sreparam/lm_head_weight_max': model.lm_head.op.weight.max().item(),
  #       'sreparam/lm_head_weight_min': model.lm_head.op.weight.min().item(),
  #       'sreparam/lm_head_weight_var': model.lm_head.op.weight.var().item(),
  #     }, step=state.global_step, commit=False)
  #     # print(model.lm_head.op.weight)