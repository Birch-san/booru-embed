def remaining_to_multiple(x: int, multiple: int) -> int:
  return (multiple - (x % multiple)) % multiple

def ceil_to_multiple(x: int, multiple: int) -> int:
  return x + remaining_to_multiple(x, multiple)