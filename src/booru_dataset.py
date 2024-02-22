from dataclasses import dataclass
from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import NamedTuple, Protocol, Optional
import numpy as np

from .vocab import Vocab

class RandomSpansNoiseMask(Protocol):
  def __call__(self, length: int) -> NDArray: ...

class BucketContent(NamedTuple):
  """
  Ragged array
  values (`NDArray`): (batch, sample) tokenized
  indices (`NDArray`): (batch+1)
  """
  values: NDArray
  indices: NDArray

class BooruDatum(NamedTuple):
  datum: NDArray
  mask_indices: NDArray

@dataclass
class BooruDataset(Dataset[BooruDatum]):
  bucket_content: BucketContent
  random_spans_noise_mask: RandomSpansNoiseMask
  # for debug (enables decoding of captions)
  vocab: Optional[Vocab] = None
  # when False, prevents masking of BOS. only suitable for padded sequences
  # (where we know BOS is at position 0).
  allow_masking_elem0: bool = False

  def __getitem__(self, index: int) -> BooruDatum:
    start, end = self.bucket_content.indices[index:index+2]
    datum: NDArray = self.bucket_content.values[start:end]
    # [self.vocab.tokens[token_ix] for token_ix in datum]

    mask_len_nominal: int = end-start
    mask_len: int = mask_len_nominal if self.allow_masking_elem0 else mask_len_nominal-1
    mask_indices_nominal: NDArray = self.random_spans_noise_mask(length=mask_len)
    if self.allow_masking_elem0:
      mask_indices: NDArray = mask_indices_nominal
    else:
      mask_indices: NDArray = np.pad(mask_indices_nominal, (1, 0))

    # if mask_indices[0]:
    #   pass
    return BooruDatum(
      datum=datum,
      mask_indices=mask_indices,
    )

  def __len__(self) -> int:
    return self.bucket_content.indices.shape[0]-1