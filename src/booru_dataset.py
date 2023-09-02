from dataclasses import dataclass
from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import NamedTuple, Protocol, Optional

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

  def __getitem__(self, index: int) -> BooruDatum:
    start, end = self.bucket_content.indices[index:index+2]
    datum: NDArray = self.bucket_content.values[start:end]
    # TODO: does random_spans_noise_mask work for lengths < 30?
    mask_indices: NDArray = self.random_spans_noise_mask(length=end-start)
    return BooruDatum(
      datum=datum,
      mask_indices=mask_indices,
    )

  def __len__(self) -> int:
    return self.bucket_content.indices.shape[0]-1