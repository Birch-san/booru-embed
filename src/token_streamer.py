from dataclasses import dataclass, field
from transformers.generation.streamers import BaseStreamer
import torch
from torch import LongTensor
from typing import List

from .vocab import Vocab

@dataclass
class TokenStreamer(BaseStreamer):
    vocab: Vocab
    batch_size: int = field(repr=False)
    acc_tok: LongTensor = field(init=False)
    decoded: List[List[str]] = field(init=False)
    def __post_init__(self):
        self.decoded = [[] for _ in range(self.batch_size)]
        self.acc_tok = torch.empty((self.batch_size, 0), dtype=torch.long)

    def put(self, value: LongTensor) -> None:
        """Function that is called by `.generate()` to push new tokens"""
        assert value.ndim == 1 or value.ndim == 2
        for acc, tok in zip(self.decoded, value[:,0] if value.ndim == 2 else value):
            acc.append(self.vocab.tokens[tok])
        self.acc_tok = torch.cat([
            self.acc_tok,
            value.unsqueeze(-1) if value.ndim == 1 else value,
        ], dim=-1)
        pass

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        # raise NotImplementedError()
        pass