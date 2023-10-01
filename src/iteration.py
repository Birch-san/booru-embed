from typing import Iterator, Iterable, TypeVar, Optional, Generator, Callable
from itertools import islice, cycle
from typing_extensions import TypeVarTuple

T = TypeVar('T')
TVT = TypeVarTuple('TVT')

def nth(iterable: Iterable[T], n: int, default: Optional[T] = None) -> Optional[T]:
	"Returns the nth item or a default value"
	return next(islice(iterable, n, None), default)

def repeatedly(iterable: Iterable[T]) -> Generator[T, None, None]:
	while True:
		it: Iterator[T] = iter(iterable)
		yield from it

def roundrobin(*iterables: Iterable[T]) -> Generator[T, None, None]:
	"roundrobin('ABC', 'D', 'EF') --> A D E B F C"
	# Recipe credited to George Sakkis
	pending: int = len(iterables)
	nexts: Iterable[Callable[[], T]] = cycle(iter(it).__next__ for it in iterables)
	while pending:
		try:
			for next in nexts:
				yield next()
		except StopIteration:
			pending -= 1
			nexts = cycle(islice(nexts, pending))