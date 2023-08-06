from typing import Iterable, TypeVar, Generator
T = TypeVar('T')

def bookend_flatten(outer_it: Iterable[Iterable[T]], bookend: T) -> Generator[T, None, None]:
  """
  This is just:
    [t for tok in general_labels for t in [*tok, comma_token_id]]
  but hopefully more efficient
  """
  for inner_it in outer_it:
    yield from inner_it
    yield bookend