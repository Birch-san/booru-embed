from re import Match
import re
from typing import List, Optional, Protocol

from .vocab import Vocab
from .booru_special_tokens import SpecialToken

class GeneralLabelToIds(Protocol):
  def __call__(self, label: str) -> List[int]: ...

qualifier_pattern = r'^(.*)_\(([^)]*?)\)$'
def get_general_label_to_ids(vocab: Vocab) -> GeneralLabelToIds:
  unk_token_id: int = vocab.token_to_ix[SpecialToken.Unknown.value]
  cosp_token_id: int = vocab.token_to_ix[SpecialToken.Cosplay.value]

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

  return general_label_to_ids