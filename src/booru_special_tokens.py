from enum import Enum
from typing import List, Literal, TypeAlias, Generator

class SpecialToken(Enum):
  # if PAD is the 0-token, it might be easier to eyeball where padding is
  Pad = '<pad>'
  EOS = '</s>'
  Unknown = '<unk>'
  CopyrightStart = '<copyright_start>'
  CharacterStart = '<character_start>'
  ArtistStart = '<artist_start>'
  MetaStart = '<meta_start>'
  GeneralStart = '<general_start>'
  EndOfGeneralLabel = ','
  # our tokenizer has a special case for splitting _(cosplay) labels, so we want to guarantee its existence
  Cosplay = 'cosplay'

Rating: TypeAlias = Literal['g', 'e', 's', 'q']
ratings: List[Rating] = ['g', 'e', 's', 'q']
def make_rating_token(rating: Rating) -> str:
  return f'rating:{rating}'

def get_booru_special_tokens(mask_token_count = 100) -> Generator[str, None, None]:
  for special_token in SpecialToken:
    yield special_token.value

  for rating in ratings:
    yield make_rating_token(rating)

  for mask_token_ix in range(mask_token_count):
    yield f'<mask_{mask_token_ix}>'
