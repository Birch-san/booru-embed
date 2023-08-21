from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import NamedTuple

class BucketContent(NamedTuple):
  """
  data (`NDArray`): (batch, sample) tokenized, padded
  lengths (`NDArray`): (batch, length)
  """
  data: NDArray
  lengths: NDArray

class BooruDatum(NamedTuple):
  datum: NDArray
  length: NDArray

class BooruDataset(Dataset[BooruDatum]):
  bucket_content: BucketContent
  def __init__(
    self,
    bucket_content: BucketContent,
  ) -> None:
    self.bucket_content = bucket_content
  
  def __getitem__(self, index: int) -> BooruDatum:
    return BooruDatum(
      datum=self.bucket_content.data[index],
      length=self.bucket_content.lengths[index],
    )

  def __len__(self) -> int:
    return self.bucket_content.lengths.shape[0]