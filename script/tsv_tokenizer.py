import fileinput
from typing import Optional, List
import torch

from src.vocab import Vocab
from src.tokenizer import make_tsv_record_to_token_ids, TsvRecordToTokenIds, Statistics

vocab = Vocab()
# create this file by running scripts/make_tokenizer.py
with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
  vocab.load(vocab_in)

statistics: Optional[Statistics] = None
tsv_record_to_token_ids: TsvRecordToTokenIds = make_tsv_record_to_token_ids(vocab, do_shuffle=False, max_tokens=255, statistics=None)

with fileinput.input(files=('/Users/birch/machine-learning/danbooru-bigquery/danbooru-captions.tsv'), encoding='utf-8') as f:
  # skip header line
  next(f)
  for line in f:
    stripped: str = line.rstrip('\n')
    token_ids: Optional[List[int]] = tsv_record_to_token_ids(stripped)
    if token_ids is None:
      continue
    print([vocab.tokens[token_id] for token_id in token_ids])
    pass
if statistics is not None:
  print(statistics.general_label_lengths)
  torch.save(torch.tensor(statistics.caption_lengths, dtype=torch.int32), 'out_analysis/caption_label_lengths.pt')
  torch.save(torch.tensor(statistics.general_label_lengths, dtype=torch.int32), 'out_analysis/general_label_lengths.pt')
pass
