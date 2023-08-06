from typing import Iterable, TypeVar, Generator
T = TypeVar('T')

def surround(it: Iterable[T], edge: T) -> Generator[T, None, None]:
  yield edge
  yield from it
  yield edge