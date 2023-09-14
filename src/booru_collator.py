from transformers import BatchEncoding
from typing import List, TypedDict, Protocol
from torch import BoolTensor, ShortTensor

from .booru_dataset import BooruDatum

class BooruCollator(Protocol):
  def __call__(self, examples: List[BooruDatum]) -> BatchEncoding: ...

class BooruBatchData(TypedDict):
    input_ids: ShortTensor
    attention_mask: BoolTensor
    labels: ShortTensor
    decoder_input_ids: ShortTensor
    decoder_attention_mask: BoolTensor