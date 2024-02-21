from os import makedirs
from os.path import join
from src.vocab import Vocab
from src.booru_special_tokens import get_booru_special_tokens
from src.map_except_first import map_except_first
from typing import Dict, Literal, List

vocab = Vocab()

for special in get_booru_special_tokens():
  vocab.add_token(special)

Category = Literal['artist', 'character', 'copyright', 'general', 'meta']
categories: List[Category] = ['artist', 'character', 'copyright', 'general', 'meta']
tokens_by_category: Dict[Category, List[str]] = {
  k: [] for k in categories
}
for category, min_prevalence in zip(
  categories,
  [128, 128, 100, 128, 100],
):
  # this file was created by:
  # - export CSV(s) via Google BigQuery (see README.md)
  # - convert to 1 tsv, via eBay tsv-utils `csv2tsv -H *.csv`
  # - running shell/split-word-prevalence-category.sh upon the tsv
  with open(f'out_prevalence_category/{category}.txt', mode='r', encoding='utf-8') as f:
    for line in f:
      stripped: str = line.rstrip('\n')
      prevalence, token = stripped.split(' ', maxsplit=1)
      if not token:
        continue
      prevalence = int(prevalence)
      if prevalence <= min_prevalence:
        break
      _, is_new = vocab.add_token(token)
      if is_new:
        tokens_by_category[category].append(token)

# check we're safe to save the dataset in int16
assert len(vocab.tokens) < (1 << 15)

out_dir = 'out_tokenizer'
makedirs(out_dir, exist_ok=True)
out_file = join(out_dir, 'vocab.txt')
with open(out_file, mode='w', encoding='utf-8') as vocab_out:
  vocab.save(vocab_out)
for category in categories:
  category_tokens: List[str] = tokens_by_category[category]
  with open(join(out_dir, f'{category}.txt'), mode='w', encoding='utf-8') as category_out:
    category_out.writelines(map_except_first(category_tokens, lambda token: '\n' + token))