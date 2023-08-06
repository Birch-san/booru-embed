from typing import List, Optional
import torch
from torch import IntTensor, bucketize, tensor
from os import makedirs
from os.path import join
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from src.vocab import Vocab
from src.tokenizer import make_tsv_record_to_token_ids, TsvRecordToTokenIds

vocab = Vocab()
# create this file by running scripts/make_tokenizer.py
with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
  vocab.load(vocab_in)

# check we're safe to save the dataset in int16
assert len(vocab.tokens) < (1 << 15)

# determined via analysis of k-means (k=20) clusters over 5.7mill Danbooru captions.
# *means* were:
# [17, 36, 55, 73, 92, 111, 129, 148, 167, 185, 204, 223, 241]
# we don't quite want means, because overflowing a bucket means shedding labels.
# we don't know the variance around each mean to know a good upper bound for each bucket
# so let's just go with the midpoint to the next bucket
#   t = torch.tensor([17, 36, 55, 73, 92, 111, 129, 148, 167, 185, 204, 223, 241], dtype=torch.int32)
#   t[:-1] + (t.diff()/2).int()
# let's round the final up to 255
buckets: IntTensor = tensor([26,  45,  64,  82, 101, 120, 138, 157, 176, 194, 213, 232, 255], dtype=torch.int16)
max_tokens: int = buckets.max().item()

tsv_record_to_token_ids: TsvRecordToTokenIds = make_tsv_record_to_token_ids(vocab, do_shuffle=True, max_tokens=max_tokens, statistics=None)

out_dir = 'out_lenbucket'
makedirs(out_dir, exist_ok=True)

value_buckets: List[List[NDArray]] = [[] for _ in buckets]
# index_buckets: List[List[int]] = [[0] for _ in buckets]
length_buckets: List[List[int]] = [[] for _ in buckets]

# this is just for testing
target_count = 100

with open('/Users/birch/machine-learning/danbooru-bigquery/danbooru-captions.tsv', mode='r', encoding='utf-8') as f:
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
    bucket_ix: int = bucketize(token_len, buckets, out_int32=True).item()

    value_bucket: List[NDArray] = value_buckets[bucket_ix]
    # index_bucket: List[int] = index_buckets[bucket_ix]
    length_bucket: List[int] = length_buckets[bucket_ix]

    token_ids_n: NDArray = np.array(token_ids, np.int16)
    value_bucket.append(token_ids_n)
    # index_bucket.append(index_bucket[-1] + token_len)
    length_bucket.append(token_len)

    target_count -= 1
    if target_count == 0:
      break
    pass

# for bucket, value_bucket, index_bucket in zip(buckets, value_buckets, index_buckets):
for bucket, value_bucket, length_bucket in zip(buckets, value_buckets, length_buckets):
  if not value_bucket:
    continue
  # assert len(index_bucket)-1 == len(value_bucket)
  assert len(length_bucket) == len(value_bucket)
  values: NDArray = np.concatenate(value_bucket, axis=-1)
  # indices: NDArray = np.array(index_bucket[:-1], dtype=np.int16)
  lengths: NDArray = np.array(length_bucket, dtype=np.int16)

  # you can compute indices from lengths. faster way to do this is to roll + assign a 0, but:
  # indices: NDArray = np.pad(lengths, (1, 0)).cumsum()[:-1]

  bucket_dir: str = join(out_dir, f'b{bucket.item()}')
  makedirs(bucket_dir, exist_ok=True)

  out_values: str = join(bucket_dir, 'values.npy')
  # out_indices: str = join(bucket_dir, 'indices.npy')
  out_lengths: str = join(bucket_dir, 'lengths.npy')
  
  np.save(out_values, values, allow_pickle=False)
  # np.save(out_indices, indices, allow_pickle=False)
  np.save(out_lengths, lengths, allow_pickle=False)
