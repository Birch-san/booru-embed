from typing import Iterable, TypeVar, Generator
T = TypeVar('T')

def intersperse_flatten(outer_it: Iterable[Iterable[T]], delimiter: T) -> Generator[T, None, None]:
  """
  This is just:
    [t for tok in general_labels for t in [*tok, comma_token_id]][:-1]
  but hopefully more efficient
  """
  first_inner_it = True
  for inner_it in outer_it:
    if not first_inner_it:
      yield delimiter
    first_inner_it = False
    yield from inner_it