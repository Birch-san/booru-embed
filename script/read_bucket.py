from os import listdir
from os.path import dirname, realpath, join
from pathlib import Path
from typing import List
import re
# import torch
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.vocab import Vocab

vocab = Vocab()
# create this file by running scripts/make_tokenizer.py
with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
  vocab.load(vocab_in)


script_dir = Path(dirname(realpath(__file__)))
repo_root: Path = script_dir.parent
in_dir = repo_root.joinpath('out_lenbucket')

potential_bucket_dirs: List[str] = listdir(in_dir)
bucket_values: List[int] = [int(dir.lstrip('b')) for dir in potential_bucket_dirs if bool(re.fullmatch(r'b[0-9]+', dir))]
bucket_values.sort()
# buckets = torch.tensor(bucket_values, dtype=torch.int16)
bucket_dirs: List[str] = [join(in_dir, f'b{val}') for val in bucket_values]

# bucket lengths:
# 21516
# 509677
# 1104165
# 1232792
# 1164978
# 834072
# 486213
# 281165
# 143312
# 70007
# 39342
# 21123
# 13500
# total=5921862
for bucket_value, bucket_dir in zip(bucket_values, bucket_dirs):
  values: NDArray = np.load(join(bucket_dir, 'values.npy'))
  lengths: NDArray = np.load(join(bucket_dir, 'lengths.npy'))
  # indices: NDArray = np.pad(lengths, (1, 0)).cumsum()[:-1]
  indices: NDArray = np.roll(lengths.cumsum(), 1)
  indices[0] = 0

  # print([vocab.tokens[token_ix] for token_ix in values[indices[0]:indices[0]+lengths[0]]])

  for index, length in tqdm(zip(indices, lengths), total=lengths.shape[-1], unit='caption'):
    caption: NDArray = values[index:index+length]
    decoded: List[str] = [vocab.tokens[token_ix] for token_ix in caption]
    # print([vocab.tokens[token_ix] for token_ix in caption])
    pass
  # break # just peeking in first bucket for now