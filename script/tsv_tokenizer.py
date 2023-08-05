import fileinput
from typing import List, Dict
import torch

from src.vocab import Vocab
from src.booru_special_tokens import make_rating_token, Rating, ratings, SpecialToken
from src.general_label_to_ids import get_general_label_to_ids, GeneralLabelToIds
from src.intersperse_flatten import intersperse_flatten

vocab = Vocab()
# create this file by running scripts/make_tokenizer.py
with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
  vocab.load(vocab_in)

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

collect_statistics = False
if collect_statistics:
  caption_lengths: List[int] = []
  general_label_lengths: List[int] = [0]*100

general_label_to_ids: GeneralLabelToIds = get_general_label_to_ids(vocab)

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
