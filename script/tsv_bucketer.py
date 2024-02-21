from typing import List, Optional, Dict, Literal
from os import makedirs
from os.path import join
from tqdm import tqdm, trange
import numpy as np
from numpy.typing import NDArray

from src.vocab import Vocab
from src.tokenizer import make_tsv_record_to_token_ids, TsvRecordToTokenIds

vocab = Vocab()
# create this file by running scripts/make_tokenizer.py
with open('out_tokenizer_2024_02/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
  vocab.load(vocab_in)

# check we're safe to save the dataset in int16
assert len(vocab.tokens) < (1 << 15)

max_tokens = 255
tsv_record_to_token_ids: TsvRecordToTokenIds = make_tsv_record_to_token_ids(vocab, do_shuffle=True, max_tokens=max_tokens, statistics=None)

out_dir = 'out_onebucket_2024_02'
makedirs(out_dir, exist_ok=True)

value_arrs: Dict[Literal['train', 'test'], List[NDArray]] = {'train': [], 'test': []}
length_arrs: Dict[Literal['train', 'test'], List[int]] = {'train': [], 'test': []}

# this is just for testing
# target_count = 100

test_split_quotient = .002 # gives us about 10k samples
train_epochs = 2

in_tsv = '/Users/birch/machine-learning/danbooru-bigquery-2024-02/danbooru-captions.tsv'

with open(in_tsv) as f:
  sample_count: int = sum(1 for _ in f)-1

train_test_cutoff = int(sample_count*(1-test_split_quotient))

for epoch in trange(0, train_epochs, unit=f'epoch', position=0):
  with open('/Users/birch/machine-learning/danbooru-bigquery-2024-02/danbooru-captions.tsv', mode='r', encoding='utf-8') as f:
    # skip header line
    next(f)
    for line_ix, line in enumerate(tqdm(f, total=sample_count, unit=f'caption', position=1)):
      stripped: str = line.rstrip('\n')
      token_ids: Optional[List[int]] = tsv_record_to_token_ids(stripped)
      if token_ids is None:
        continue
      # print([vocab.tokens[token_id] for token_id in token_ids])

      train_test_key = 'test' if line_ix > train_test_cutoff else 'train'
      if train_test_key == 'test' and epoch > 0:
        # we don't need multiple epochs for test data
        break

      value_arr: List[NDArray] = value_arrs[train_test_key]
      length_arr: List[int] = length_arrs[train_test_key]
      
      token_len: int = len(token_ids)

      token_ids_n: NDArray = np.array(token_ids, np.int16)
      value_arr.append(token_ids_n)
      length_arr.append(token_len)

      # target_count -= 1
      # if target_count == 0:
      #   break
      pass

for split in ['train', 'test']:
  print(f'saving split {split}..')
  value_arr: List[NDArray] = value_arrs[train_test_key]
  length_arr: List[int] = length_arrs[train_test_key]
  assert len(length_arr) == len(value_arr)
  values: NDArray = np.concatenate(value_arr, axis=-1)
  lengths: NDArray = np.array(length_arr, dtype=np.int16)

  # compute indices from lengths
  indices: NDArray = np.pad(lengths, (1, 0)).cumsum(dtype=np.int32)

  split_dir: str = join(out_dir, split)
  makedirs(split_dir, exist_ok=True)

  out_values: str = join(split_dir, 'values.npy')
  out_indices: str = join(split_dir, 'indices.npy')
  out_lengths: str = join(split_dir, 'lengths.npy')
  
  np.save(out_values, values, allow_pickle=False)
  np.save(out_indices, indices, allow_pickle=False)
  np.save(out_lengths, lengths, allow_pickle=False)
