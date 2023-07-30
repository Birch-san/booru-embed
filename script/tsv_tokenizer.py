import fileinput
from dataclasses import dataclass, field
from typing import List, Set, Literal, Dict

@dataclass
class Vocab:
  token_to_ix: Dict[str, int] = field(default_factory=lambda:{})
  tokens: List[str] = field(default_factory=lambda:[])
  def add_token(self, token: str) -> int:
    if token in self.token_to_ix:
      return self.token_to_ix[token]
    token_ix: int = len(self.tokens)
    self.tokens.append(token)
    self.token_to_ix[token] = token_ix

vocab = Vocab()

for special_token in [
  'BOS',
  'EOS',
  'PAD',
  'MASK',
  'COPYRIGHT_START',
  'CHARACTER_START',
  'ARTIST_START',
  'META_START',
  'GENERAL_START',
  ',',
]:
  vocab.add_token(special_token)

for rating in ['g', 'e', 's', 'q']:
  vocab.add_token(f'rating:{rating}')

for category, min_prevalence in zip(
  ['artist', 'character', 'copyright', 'general', 'meta'],
  [100, 128, 100, 128, 100],
):
  with fileinput.input(files=(f'out_prevalence_category/{category}.txt'), encoding='utf-8') as f:
    for line in f:
      stripped: str = line.rstrip('\n')
      prevalence, token = stripped.split(' ', maxsplit=1)
      prevalence = int(prevalence)
      if prevalence <= min_prevalence:
        break
      vocab.add_token(token)

with fileinput.input(files=('/Users/birch/machine-learning/danbooru-bigquery/danbooru-captions.tsv'), encoding='utf-8') as f:
  # skip header line
  next(f)
  for line in f:
    stripped: str = line.rstrip('\n')
    tab_split: List[str] = stripped.split('\t')
    id, rating, score, meta, general, artist, copyright, character = tab_split
    id = int(id)
    rating: Literal['g', 'e', 's', 'q'] = rating
    pass
