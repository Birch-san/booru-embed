from dataclasses import dataclass
from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import NamedTuple, Protocol

class RandomSpansNoiseMask(Protocol):
  def __call__(self, length: int) -> NDArray: ...

class BucketContent(NamedTuple):
  """
  Ragged array
  values (`NDArray`): (batch, sample) tokenized
  indices_and_lengths (`NDArray`): (batch, 2) each row is an [index, length] 2-tuple
  """
  values: NDArray
  indices_and_lengths: NDArray

class BooruDatum(NamedTuple):
  datum: NDArray
  mask_indices: NDArray

@dataclass
class BooruDataset(Dataset[BooruDatum]):
  bucket_content: BucketContent
  random_spans_noise_mask: RandomSpansNoiseMask

  def __getitem__(self, index: int) -> BooruDatum:
    index_and_length: NDArray = self.bucket_content.indices_and_lengths[index]
    index, length = index_and_length
    datum: NDArray = self.bucket_content.values[index:index+length]
    mask_indices: NDArray = self.random_spans_noise_mask(length=length)
    return BooruDatum(
      datum=datum,
      mask_indices=mask_indices,
    )

  def __len__(self) -> int:
    return self.bucket_content.indices_and_lengths.shape[0]