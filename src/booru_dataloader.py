from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t
from typing import Generic, Iterable, List

class BooruDataLoader(DataLoader[T_co], Generic[T_co]):
  def __init__(
    self,
    dataset: Dataset[T_co],
    batch_size: int | None = 1,
    shuffle: bool | None = None,
    sampler: Sampler | Iterable | None = None,
    batch_sampler: Sampler[List] | Iterable[List] | None = None,
    num_workers: int = 0,
    collate_fn: _collate_fn_t | None = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: _worker_init_fn_t | None = None,
    multiprocessing_context=None,
    generator=None,
    *args,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
  ):
    super().__init__(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      sampler=sampler,
      batch_sampler=batch_sampler,
      num_workers=num_workers,
      collate_fn=collate_fn,
      pin_memory=pin_memory,
      drop_last=drop_last,
      timeout=timeout,
      worker_init_fn=worker_init_fn,
      multiprocessing_context=multiprocessing_context,
      generator=generator,
      *args,
      prefetch_factor=prefetch_factor,
      persistent_workers=persistent_workers,
      pin_memory_device=pin_memory_device,
    )
  
  def __len__(self) -> int:
    return super().__len__()