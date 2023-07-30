import fileinput
from dataclasses import dataclass, field
from typing import List, Set, Literal, Dict, TypeAlias
from enum import Enum

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

class SpecialToken(Enum):
  BOS = 'BOS'
  EOS = 'EOS'
  Pad = 'PAD'
  Unknown = 'UNK'
  Mask = 'MASK'
  CopyrightStart = 'COPYRIGHT_START'
  CharacterStart = 'CHARACTER_START'
  ArtistStart = 'ARTIST_START'
  MetaStart = 'META_START'
  GeneralStart = 'GENERAL_START'
  Comma = ','

Rating: TypeAlias = Literal['g', 'e', 's', 'q']
ratings: List[Rating] = ['g', 'e', 's', 'q']
def make_rating_token(rating: Rating) -> str:
  return f'rating:{rating}'

for special_token in SpecialToken:
  vocab.add_token(special_token.value)

for rating in ratings:
  vocab.add_token(make_rating_token(rating))

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

# micro-optimization to avoid dict lookup of tokens we'll be using often
bos_token_id: int = vocab.token_to_ix[SpecialToken.BOS.value]
eos_token_id: int = vocab.token_to_ix[SpecialToken.EOS.value]
char_token_id: int = vocab.token_to_ix[SpecialToken.CharacterStart.value]
cpy_token_id: int = vocab.token_to_ix[SpecialToken.CopyrightStart.value]
unk_token_id: int = vocab.token_to_ix[SpecialToken.Unknown.value]
art_token_id: int = vocab.token_to_ix[SpecialToken.ArtistStart.value]
meta_token_id: int = vocab.token_to_ix[SpecialToken.MetaStart.value]
gen_token_id: int = vocab.token_to_ix[SpecialToken.GeneralStart.value]
comma_token_id: int = vocab.token_to_ix[SpecialToken.Comma.value]

# micro-optimization to, uh, look up from a smaller dict (might use linear probing rather than hash?)
rating_token_ids: Dict[Rating, int] = {
  rating: vocab.token_to_ix[make_rating_token(rating)] for rating in ratings
}

with fileinput.input(files=('/Users/birch/machine-learning/danbooru-bigquery/danbooru-captions.tsv'), encoding='utf-8') as f:
  # skip header line
  next(f)
  for line in f:
    stripped: str = line.rstrip('\n')
    tab_split: List[str] = stripped.split('\t')
    id, rating, score, meta, general, artist, copyright, character = tab_split
    id = int(id)
    rating: Rating = rating
    meta_token_ids: List[int] = [vocab.token_to_ix.get(tok, unk_token_id) for tok in meta.split(' ')]
    # TODO
    tokens: List[int] = [
      bos_token_id,
      rating_token_ids[rating],
      char_token_id,
      cpy_token_id,
      art_token_id,
      gen_token_id,
      meta_token_id,
      eos_token_id,
    ]
    pass
