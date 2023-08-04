import fileinput
from dataclasses import dataclass, field
from typing import List, Literal, Dict, TypeAlias, Optional, Generator, Iterable, TypeVar
from enum import Enum
import re
from re import Match
import torch
from torch import IntTensor, bucketize, tensor
from contextlib import ExitStack
from os import makedirs
from os.path import join
from webdataset import ShardWriter
from tqdm import tqdm

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
  # if PAD is the 0-token, it might be easier to eyeball where padding is
  Pad = '<pad>'
  EOS = '</s>'
  Unknown = '<unk>'
  CopyrightStart = '<copyright_start>'
  CharacterStart = '<character_start>'
  ArtistStart = '<artist_start>'
  MetaStart = '<meta_start>'
  GeneralStart = '<general_start>'
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

mask_token_count = 100
for mask_token_ix in range(mask_token_count):
  # start with 99, go down to 0 inclusive
  decr_ix: int = (mask_token_count-1)-mask_token_ix
  vocab.add_token(f'<extra_id_{mask_token_ix}>')

# micro-optimization to avoid dict lookup of tokens we'll be using often
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

# determined via analysis of k-means (k=20) clusters over 5.7mill Danbooru captions.
# *means* were:
# [14, 33, 52, 70, 89, 108, 126, 145, 164, 182, 201, 220, 238]
# we don't quite want means, because overflowing a bucket means shedding labels.
# we don't know the variance around each mean to know a good upper bound for each bucket
# so let's just go with the midpoint to the next bucket
#   t = torch.tensor([14, 33, 52, 70, 89, 108, 126, 145, 164, 182, 201, 220, 238], dtype=torch.int32)
#   t[:-1] + (t.diff()/2).int()
# let's subtract 1 from all of those because I removed BOS from every prompt after computing these
# let's also add a 255 bucket, for the upper bound we need to support to cover 99.75% of Danbooru 
buckets: IntTensor = tensor([22, 41, 60, 78, 97, 116, 134, 153, 172, 190, 209, 228, 245, 255], dtype=torch.int32)
max_tokens: int = buckets.max().item()

out_dir = '/home/birch/ml-data/booru-captions-out-lenbucket'
makedirs(out_dir, exist_ok=True)
bucket_dirnames: List[str] = [f'bucket_{bucket}' for bucket in buckets]
bucket_dirs: List[str] = [join(out_dir, name) for name in bucket_dirnames]
print(f"making bucket dirs under {out_dir}: {bucket_dirnames}")
for bucket_dir in bucket_dirs:
  makedirs(bucket_dir, exist_ok=True)

with ExitStack() as stack:
  # TODO: for our fixed-length tensor data, it might be better to use a pandas dataframe than a webdataset
  shard_writers: List[ShardWriter] = [
    ShardWriter(join(bucket_out, '%05d.tar'),maxcount=10000) for bucket_out in bucket_dirs
  ]
  for mgr in shard_writers:
    stack.enter_context(mgr)

  with fileinput.input(files=('/Users/birch/machine-learning/danbooru-bigquery/danbooru-captions.tsv'), encoding='utf-8') as f:
    # skip header line
    next(f)
    samples_estimate = 5770089
    for line in tqdm(f, total=samples_estimate, unit=f'caption'):
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
      # print([vocab.tokens[token_ix] for token_ix in token_ixs])

      token_len: int = len(token_ixs)
      if token_len > max_tokens:
        # I mean we could drop labels to salvage it, but probably too many subjects are being portrayed to get a good embedding anyway
        continue
      # TODO: shuffle
      # TODO: pad to bucket length

      bucket_ix: int = bucketize(token_len, buckets).item()

      sink: ShardWriter = shard_writers[bucket_ix]
      pass
  pass
