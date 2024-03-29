from enum import Enum
from typing import List, Literal, TypeAlias, Generator

class SpecialToken(Enum):
  # if PAD is the 0-token, it might be easier to eyeball where padding is
  Pad = '<pad>'
  # in order to be able to employ chunking, we use a per-sequence BOS but a single EOS to delimit prompt from continuation
  BOS = '<s>'
  EOS = '</s>'
  Unknown = '<unk>'
  ConvPad = '<cpad>'
  CopyrightStart = '<copyright_start>'
  CharacterStart = '<character_start>'
  ArtistStart = '<artist_start>'
  MetaStart = '<meta_start>'
  GeneralStart = '<general_start>'
  EdgeOfGeneralLabel = ','
  # our tokenizer has a special case for splitting _(cosplay) labels, so we want to guarantee its existence
  Cosplay = 'cosplay'

Rating: TypeAlias = Literal['g', 'e', 's', 'q']
ratings: List[Rating] = ['g', 'e', 's', 'q']
def make_rating_token(rating: Rating) -> str:
  return f'rating:{rating}'

def make_mask_token(ix: int) -> str:
  return f'<mask_{ix}>'

def make_vocab_pad_token(ix: int) -> str:
  """
  For padding the vocab length to a multiple of 8, to optimize for tensor cores
  https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf
  """
  return f'<i_love_tensor_cores_{ix}>'

def get_booru_special_tokens(mask_token_count = 100) -> Generator[str, None, None]:
  for special_token in SpecialToken:
    yield special_token.value

  for rating in ratings:
    yield make_rating_token(rating)

  for mask_token_ix in range(mask_token_count):
    yield make_mask_token(mask_token_ix)
