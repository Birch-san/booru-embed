import fileinput
from typing import List, Dict, Any
import torch
from torch import IntTensor, bucketize, tensor
from contextlib import ExitStack
from os import makedirs
from os.path import join
from tqdm import tqdm
from random import shuffle
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
import numpy as np

from src.vocab import Vocab
from src.booru_special_tokens import make_rating_token, Rating, ratings, SpecialToken
from src.general_label_to_ids import get_general_label_to_ids, GeneralLabelToIds
from src.intersperse_flatten import intersperse_flatten

vocab = Vocab()
# create this file by running scripts/make_tokenizer.py
with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
  vocab.load(vocab_in)

schema = pa.schema([
  ('input_ids', pa.int16()),
])

# check we're safe to save the dataset in int16
assert len(vocab.tokens) < (1 << 15)

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

# out_dir = '/home/birch/ml-data/booru-captions-out-lenbucket'
out_dir = 'out_lenbucket'
makedirs(out_dir, exist_ok=True)
bucket_filenames: List[str] = [f'bucket_{bucket}.parquet' for bucket in buckets]
bucket_paths: List[str] = [join(out_dir, fname) for fname in bucket_filenames]
# bucket_dirnames: List[str] = [f'bucket_{bucket}' for bucket in buckets]
# bucket_dirs: List[str] = [join(out_dir, name) for name in bucket_dirnames]
# print(f"making bucket dirs under {out_dir}: {bucket_dirnames}")
# for bucket_dir in bucket_dirs:
#   makedirs(bucket_dir, exist_ok=True)

def shuffle_(l: List[Any]) -> None:
  if len(l) >= 2:
    shuffle(l)

general_label_to_ids: GeneralLabelToIds = get_general_label_to_ids(vocab)

with ExitStack() as stack:
  parquet_writers: List[ParquetWriter] = [
    ParquetWriter(bucket_path, schema=schema) for bucket_path in bucket_paths
  ]
  for mgr in parquet_writers:
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

      general_token_ids_len: int = sum((len(x) for x in general_labels)) + len(general_labels)-1
      # compute length first, as a fast-path to discard long prompts before we commit to the cost of shuffling
      token_len: int = 7 + len(char_token_ids) + len(cpy_token_ids) + len(art_token_ids) + general_token_ids_len + len(meta_token_ids)
      if token_len > max_tokens:
        # I mean we could drop labels to salvage it, but probably too many subjects are being portrayed to get a good embedding anyway
        continue

      shuffle_(char_token_ids)
      shuffle_(cpy_token_ids)
      shuffle_(art_token_ids)
      shuffle_(general_labels)
      shuffle_(meta_token_ids)

      general_token_ids: List[int] = list(intersperse_flatten(general_labels, comma_token_id))
      assert len(general_token_ids) == general_token_ids_len

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
      assert len(token_ixs) == token_len
      
      bucket_ix: int = bucketize(token_len, buckets).item()

      sink: ParquetWriter = parquet_writers[bucket_ix]
      arr = pa.array(np.array(token_ixs, np.int16), pa.int16(), mask=None, size=token_len)
      table = pa.Table.from_arrays([arr], schema=schema)
      sink.write_table(table)
      pass
  pass
