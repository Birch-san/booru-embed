from typing import Protocol, TypeVar, Generic

class Test(Protocol):
  @staticmethod
  def __call__(x: str) -> int: ...

OutShape = TypeVar('OutShape')

class FlopCustomMapping(Protocol, Generic[OutShape]):
  @staticmethod
  def __call__(*args, out_shape: OutShape = (), **kwargs) -> int: ...