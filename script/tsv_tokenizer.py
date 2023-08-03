import fileinput
from dataclasses import dataclass, field
from typing import List, Literal, Dict, TypeAlias, Optional, Generator, Iterable, Iterator, TypeVar
from enum import Enum
import re
from re import Match
import torch

T = TypeVar('T')

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
  # our tokenizer has a special case for splitting _(cosplay) labels, so we want to guarantee its existence
  Cosplay = 'cosplay'

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
cosp_token_id: int = vocab.token_to_ix[SpecialToken.Cosplay.value]

# micro-optimization to, uh, look up from a smaller dict (might use linear probing rather than hash?)
rating_token_ids: Dict[Rating, int] = {
  rating: vocab.token_to_ix[make_rating_token(rating)] for rating in ratings
}

qualifier_pattern = r'^(.*)_\(([^)]*?)\)$'
def general_label_to_ids(label: str) -> List[id]:
  # prefer whole-label match if possible
  if label in vocab.token_to_ix:
    return [vocab.token_to_ix[label]]
  # we don't split short labels, because they're likely to be kaomoji
  if len(label) <= 4:
    return [unk_token_id]
  
  qualifier_ids: List[int] = []
  rest: str = label
  m: Optional[Match[str]] = None
  while m := re.search(qualifier_pattern, rest):
    rest = m.group(1)
    qualifier: str = m.group(2)
    if qualifier == 'cosplay':
      name_id: int = vocab.token_to_ix.get(rest, unk_token_id)
      # _(cosplay) qualifies a name label, so what precedes it can be used
      # without further splitting
      return [
        name_id,
        cosp_token_id,
        # when I did the token analysis I hadn't considered possibility that
        # any qualifier could appear *after* _(cosplay), but if it did I suppose
        # we should treat everything before cosplay as a name and everything after
        # as per-qualifier tokens
        *qualifier_ids,
      ]
    qualifier_ids.append(vocab.token_to_ix.get(qualifier, unk_token_id))

  words: List[str] = re.split(r'[-_]', rest)
  return [
    *[vocab.token_to_ix.get(word, unk_token_id) for word in words],
    *qualifier_ids,
  ]

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

collect_statistics = False
if collect_statistics:
  caption_lengths: List[int] = []
  general_label_lengths: List[int] = [0]*100

with fileinput.input(files=('/Users/birch/machine-learning/danbooru-bigquery/danbooru-captions.tsv'), encoding='utf-8') as f:
  # skip header line
  next(f)
  for line in f:
    stripped: str = line.rstrip('\n')
    tab_split: List[str] = stripped.split('\t')
    id, rating, score, meta, general, artist, copyright, character = tab_split
    id = int(id)
    rating: Rating = rating
    # fallback to UNK for unknown character may be fine because we can correlate character with copyright and hair colour
    char_token_ids: List[int] = [vocab.token_to_ix.get(tok, unk_token_id) for tok in character.split(' ') if tok in vocab.token_to_ix]
    # fallback to UNK for copyright may be fine because we can correlate copyright with characters, or general labels
    cpy_token_ids: List[int] = [vocab.token_to_ix.get(tok, unk_token_id) for tok in copyright.split(' ') if tok in vocab.token_to_ix]
    # fallback to UNK for artist is a bit tenuous, but at least helps create an at-least-one-artist pattern, to help predict end-of-artist section
    art_token_ids: List[int] = [vocab.token_to_ix.get(tok, unk_token_id) for tok in artist.split(' ') if tok in vocab.token_to_ix]
    # probably not helpful to fallback to UNK for meta tokens, because they're not so correlated with other labels
    meta_token_ids: List[int] = [vocab.token_to_ix.get(tok) for tok in meta.split(' ') if tok in vocab.token_to_ix]
    general_labels: List[List[int]] = [general_label_to_ids(tok) for tok in general.split(' ')]
    general_token_ids: List[int] = list(intersperse_flatten(general_labels, comma_token_id))

    token_ixs: List[int] = [
      bos_token_id,
      rating_token_ids[rating],
      char_token_id,
      *char_token_ids,
      cpy_token_id,
      *cpy_token_ids,
      art_token_id,
      *art_token_ids,
      gen_token_id,
      *general_token_ids,
      meta_token_id,
      *meta_token_ids,
      eos_token_id,
    ]
    # TODO: shuffle
    # print([vocab.tokens[token_ix] for token_ix in token_ixs])
    if collect_statistics:
      for general_label in general_labels:
        general_label_lengths[len(general_label)] += 1
      caption_lengths.append(len(token_ixs))
    pass
if collect_statistics:
  print(general_label_lengths)
  torch.save(torch.tensor(caption_lengths, dtype=torch.int32), 'out_analysis/caption_label_lengths.pt')
  torch.save(torch.tensor(general_label_lengths, dtype=torch.int32), 'out_analysis/general_label_lengths.pt')
pass
