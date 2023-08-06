from typing import Iterable, TypeVar, Generator, Callable
T = TypeVar('T')

def map_except_first(it: Iterable[T], mapper: Callable[[T], T]) -> Generator[T, None, None]:
  first_elem = True
  # note: it would probably be more efficient to do a next() and then a loop
  # than to evaluate first_elem repeatedly. but handling the empty iterable case would be a bit ugly
  # and I reckon branch-prediction will eliminate the cost of the repeated condition anyway
  for elem in it:
    if first_elem:
      first_elem = False
      yield elem
    else:
      yield mapper(elem)