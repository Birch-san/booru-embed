import fileinput
from typing import List, Optional
import torch
from torch import IntTensor, bucketize, tensor
from contextlib import ExitStack
from os import makedirs
from os.path import join
from tqdm import tqdm
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
import numpy as np

from src.vocab import Vocab
from src.tokenizer import make_tsv_record_to_token_ids, TsvRecordToTokenIds

vocab = Vocab()
# create this file by running scripts/make_tokenizer.py
with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
  vocab.load(vocab_in)

schema = pa.schema([
  ('input_ids', pa.int16()),
])

# check we're safe to save the dataset in int16
assert len(vocab.tokens) < (1 << 15)

# determined via analysis of k-means (k=20) clusters over 5.7mill Danbooru captions.
# *means* were:
# [14, 33, 52, 70, 89, 108, 126, 145, 164, 182, 201, 220, 238]
# we don't quite want means, because overflowing a bucket means shedding labels.
# we don't know the variance around each mean to know a good upper bound for each bucket
# so let's just go with the midpoint to the next bucket
#   t = torch.tensor([14, 33, 52, 70, 89, 108, 126, 145, 164, 182, 201, 220, 238], dtype=torch.int32)
#   t[:-1] + (t.diff()/2).int()
# let's also add a 255 bucket, for the upper bound we need to support to cover 99.75% of Danbooru 
buckets: IntTensor = tensor([23, 42, 61, 79, 98, 117, 135, 154, 173, 191, 210, 229, 246, 255], dtype=torch.int32)
max_tokens: int = buckets.max().item()

tsv_record_to_token_ids: TsvRecordToTokenIds = make_tsv_record_to_token_ids(vocab, do_shuffle=True, max_tokens=max_tokens, statistics=None)

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
      token_ids: Optional[List[int]] = tsv_record_to_token_ids(stripped)
      if token_ids is None:
        continue
      # print([vocab.tokens[token_id] for token_id in token_ids])
      
      token_len: int = len(token_ids)
      bucket_ix: int = bucketize(token_len, buckets).item()

      sink: ParquetWriter = parquet_writers[bucket_ix]
      arr = pa.array(np.array(token_ids, np.int16), pa.int16(), mask=None, size=token_len)
      table = pa.Table.from_arrays([arr], schema=schema)
      sink.write_table(table)
      pass
  pass
