from dataclasses import dataclass
from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import NamedTuple, Protocol
import numpy as np

class RandomSpansNoiseMask(Protocol):
  def __call__(self, length: int) -> NDArray: ...

# class PadArray(Protocol):
#   def __call__(self, shape: int) -> NDArray: ...

class BucketContent(NamedTuple):
  """
  data (`NDArray`): (batch, sample) tokenized, padded
  lengths (`NDArray`): (batch, length)
  """
  data: NDArray
  lengths: NDArray

class BooruDatum(NamedTuple):
  datum: NDArray
  length: np.int16
  mask_indices: NDArray

@dataclass
class BooruDataset(Dataset[BooruDatum]):
  bucket_content: BucketContent
  random_spans_noise_mask: RandomSpansNoiseMask
  eos_token_id: int

  # def __postinit__(self):
  #   if self.eos_token_id == 0:
  #     # we can use ndarray#resize to pad more cheaply
  #     pass

  def __getitem__(self, index: int) -> BooruDatum:
    length: np.int16 = self.bucket_content.lengths[index]
    # padded_mask_indices: NDArray = 
    mask_indices: NDArray = self.random_spans_noise_mask(length=length)
    return BooruDatum(
      datum=self.bucket_content.data[index],
      length=self.bucket_content.lengths[index],
      mask_indices=mask_indices,
    )

  def __len__(self) -> int:
    return self.bucket_content.lengths.shape[0]