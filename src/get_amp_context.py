import torch
from contextlib import nullcontext
from logging import getLogger
from transformers import TrainingArguments

logger = getLogger(__name__)

def get_amp_context(args: TrainingArguments) -> torch.cuda.amp.autocast | nullcontext:
    match args.half_precision_backend:
        case 'auto' | 'cuda':
          amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
          return torch.cuda.amp.autocast(cache_enabled=True, dtype=amp_dtype)
        case _:
          logger.warning(f'half_precision_backend={args.half_precision_backend} not implemented; disabling amp')
          return nullcontext()