from transformers import BatchEncoding
from dataclasses import dataclass, field
from typing import List, Optional
from .booru_dataset import BooruDatum
from .booru_collator import BooruCollator

@dataclass
class BooruReplayCollator(BooruCollator):
  collator: BooruCollator
  cached_batch: Optional[BatchEncoding] = field(default=None, init=False)
  def __call__(self, examples: List[BooruDatum]) -> BatchEncoding:
    if self.cached_batch is None:
      self.cached_batch = self.collator(examples)
    return self.cached_batch