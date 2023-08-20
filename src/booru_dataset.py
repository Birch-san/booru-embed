from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import NamedTuple

class BooruDatum(NamedTuple):
  datum: NDArray
  length: NDArray

class BooruDataset(Dataset[BooruDatum]):
  data: NDArray
  lengths: NDArray
  def __init__(
    self,
    data: NDArray,
    lengths: NDArray,
  ) -> None:
    """
    Args
      data (`NDArray`): (batch, sample) tokenized, padded
      lengths (`NDArray`): (batch, length)
    """
    self.data = data
    self.lengths = lengths
  
  def __getitem__(self, index: int) -> BooruDatum:
    return BooruDatum(
      datum=self.data[index],
      length=self.lengths[index],
    )

  def __len__(self) -> int:
    return self.lengths.size(0)