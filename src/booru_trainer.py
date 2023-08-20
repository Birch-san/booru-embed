import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Dict, List, Optional, Tuple, Any

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from .booru_dataloader import BooruDataLoader

class BooruTrainer(Trainer):
  """
  Customizes `DataLoader` to try and avoid pyarrow, in favour of just seeking into big numpy matrices
  """
  def __init__(
    self,
    model: PreTrainedModel | Module = None,
    args: TrainingArguments = None,
    data_collator: Any | None = None,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | Dict[str, Dataset] | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model_init: Callable[[], PreTrainedModel] | None = None,
    compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
    callbacks: List[TrainerCallback] | None = None, optimizers: Tuple[Optimizer, LambdaLR] = ...,
    preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None,
  ):
    super().__init__(
      model=model,
      args=args,
      data_collator=data_collator,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      tokenizer=tokenizer,
      model_init=model_init,
      compute_metrics=compute_metrics,
      callbacks=callbacks,
      optimizers=optimizers,
      preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
  
  def num_examples(self, dataloader: DataLoader) -> int:
    return super().num_examples(dataloader=dataloader)

  def get_train_dataloader(self) -> BooruDataLoader:
    return super().get_train_dataloader()

  def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> BooruDataLoader:
    return super().get_eval_dataloader(eval_dataset=eval_dataset)