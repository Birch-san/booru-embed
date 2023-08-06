from dataclasses import dataclass, field
from typing import List, Protocol, Dict, Any, Optional
from random import shuffle

from .vocab import Vocab
from .booru_special_tokens import Rating, SpecialToken, make_rating_token, ratings
from .general_label_to_ids import GeneralLabelToIds, get_general_label_to_ids
from .bookend_flatten import bookend_flatten

class TsvRecordToTokenIds(Protocol):
  def __call__(self, tabdelim: str) -> List[int]: ...

def shuffle_(l: List[Any]) -> None:
  if len(l) >= 2:
    shuffle(l)

@dataclass
class Statistics:
  caption_lengths: List[int] = field(default_factory=lambda: [])
  general_label_lengths: List[int] = field(default_factory=lambda: [0]*100)

def make_tsv_record_to_token_ids(
  vocab: Vocab,
  do_shuffle: bool,
  max_tokens: int,
  statistics: Optional[Statistics] = None,
) -> TsvRecordToTokenIds:
  # micro-optimization to avoid dict lookup of tokens we'll be using often
  eos_token_id: int = vocab.token_to_ix[SpecialToken.EOS.value]
  cnv_token_id: int = vocab.token_to_ix[SpecialToken.ConvPad.value]
  char_token_id: int = vocab.token_to_ix[SpecialToken.CharacterStart.value]
  cpy_token_id: int = vocab.token_to_ix[SpecialToken.CopyrightStart.value]
  unk_token_id: int = vocab.token_to_ix[SpecialToken.Unknown.value]
  art_token_id: int = vocab.token_to_ix[SpecialToken.ArtistStart.value]
  meta_token_id: int = vocab.token_to_ix[SpecialToken.MetaStart.value]
  gen_token_id: int = vocab.token_to_ix[SpecialToken.GeneralStart.value]
  comma_token_id: int = vocab.token_to_ix[SpecialToken.EdgeOfGeneralLabel.value]

  # micro-optimization to, uh, look up from a smaller dict (might use linear probing rather than hash?)
  rating_token_ids: Dict[Rating, int] = {
    rating: vocab.token_to_ix[make_rating_token(rating)] for rating in ratings
  }
  
  general_label_to_ids: GeneralLabelToIds = get_general_label_to_ids(vocab)

  def tsv_record_to_token_ids(tabdelim: str) -> Optional[List[int]]:
    tab_split: List[str] = tabdelim.split('\t')
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

    general_token_ids_len: int = sum((len(x) for x in general_labels)) + len(general_labels)
    # compute length first, as a fast-path to discard long prompts before we commit to the cost of shuffling
    token_len: int = 10 + len(char_token_ids) + len(cpy_token_ids) + len(art_token_ids) + general_token_ids_len + len(meta_token_ids)
    if token_len > max_tokens:
      # I mean we could drop labels to salvage it, but probably too many subjects are being portrayed to get a good embedding anyway
      return None

    if do_shuffle:
      shuffle_(char_token_ids)
      shuffle_(cpy_token_ids)
      shuffle_(art_token_ids)
      shuffle_(general_labels)
      shuffle_(meta_token_ids)

    general_token_ids: List[int] = list(bookend_flatten(general_labels, comma_token_id))
    assert len(general_token_ids) == general_token_ids_len

    token_ixs: List[int] = [
      cnv_token_id,
      rating_token_ids[rating],
      char_token_id,
      *char_token_ids,
      cpy_token_id,
      *cpy_token_ids,
      art_token_id,
      *art_token_ids,
      gen_token_id,
      comma_token_id,
      *general_token_ids,
      meta_token_id,
      *meta_token_ids,
      eos_token_id,
      cnv_token_id,
    ]
    # print([vocab.tokens[token_ix] for token_ix in token_ixs])
    assert len(token_ixs) == token_len

    if statistics is not None:
      for general_label in general_labels:
        statistics.general_label_lengths[len(general_label)] += 1
      statistics.caption_lengths.append(len(token_ixs))
    
    return token_ixs
  
  return tsv_record_to_token_ids