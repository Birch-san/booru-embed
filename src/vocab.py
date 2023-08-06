from dataclasses import dataclass, field
from typing import Dict, List
from io import TextIOWrapper
from .map_except_first import map_except_first

@dataclass
class Vocab:
  token_to_ix: Dict[str, int] = field(default_factory=lambda:{})
  tokens: List[str] = field(default_factory=lambda:[])
  def add_token(self, token: str) -> int:
    if not token:
      raise ValueError('empty tokens are no bueno')
    if token in self.token_to_ix:
      return self.token_to_ix[token]
    token_ix: int = len(self.tokens)
    self.tokens.append(token)
    self.token_to_ix[token] = token_ix
  
  def save(self, file: TextIOWrapper) -> None:
    """
    vocab = Vocab()
    vocab.add_token('hello')
    with open('vocab.txt', mode='w', encoding='utf-8') as vocab_out:
      vocab.save(vocab_out)
    """
    file.writelines(map_except_first(self.tokens, lambda token: '\n' + token))
  
  def load(self, file: TextIOWrapper) -> None:
    """
    vocab = Vocab()
    with open('vocab.txt', mode='r', encoding='utf-8') as vocab_in:
      vocab.load(vocab_in)
    """
    for line in file:
      stripped: str = line.rstrip('\n')
      self.add_token(stripped)

