from dataclasses import dataclass, field
from transformers import BatchEncoding
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
import torch
from torch import ByteTensor, BoolTensor, ShortTensor, full

from .booru_dataset import BooruDatum
from .vocab import Vocab
from .ceil_to_multiple import ceil_to_multiple
from .booru_collator import BooruCollator, BooruBatchData

@dataclass
class BooruDataCollatorForT5MLM(BooruCollator):
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        eos_token_id (:obj:`int`)
        sentinel_start_ix (:obj:`int`)
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    eos_token_id: int
    sentinel_start_ix: int
    pad_token_id: int
    label_ignore_index: int
    decoder_start_token_id: int
    # for debug (enables decoding of captions)
    vocab: Optional[Vocab] = None
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    # xformers kernels only support attention bias for sequence lengths multiple of 8
    pad_to_multiple: Optional[int] = None
    pad_to_max: bool = False
    max_length: Optional[int] = None

    def __post_init__(self):
        assert self.pad_token_id == 0, 'we currently employ in filter_input_ids a condition which ignores both -1 tokens and pad tokens via the criteria `<= 0` (on the basis it may be cheaper than `!= -1 & != self.pad_token_id`). to use a non-zero pad_token_id: we would need to remove this optimization.'
        if self.pad_to_max:
            assert self.max_length is not None, 'You have requested pad_to_max, but you have not specified a max_length'
            assert self.max_length % self.pad_to_multiple == 0, f'You have requested pad_to_multiple={self.pad_to_multiple}, and pad_to_max=True. This requires your max_length={self.max_length} to be a multiple of pad_to_multiple={self.pad_to_multiple}.'
            self.pad_to_multiple = self.max_length

    def __call__(self, examples: List[BooruDatum]) -> BatchEncoding:
        # tensorize input
        lengths: NDArray = np.hstack([e[0].shape[0] for e in examples], dtype=np.int16)
        max_len: np.int16 = lengths.max()

        input_ids: NDArray = np.full((len(examples), max_len), fill_value=self.pad_token_id, dtype=np.int16)
        mask_indices: NDArray = np.full((len(examples), max_len), fill_value=False, dtype=np.bool_)
        for ix, (caption, mask_indices_) in enumerate(examples):
            caption_len = caption.shape[0]
            input_ids[ix, :caption_len] = caption
            mask_indices[ix, :caption_len] = mask_indices_
        
        # [[self.vocab.tokens[token_ix] for token_ix in caption] for caption in input_ids]

        mask_indices_t: BoolTensor = torch.from_numpy(mask_indices).to(self.device, torch.int8)
        # TODO: these can be done in parallel. try non_blocking=True
        input_ids_sentinel_t: ByteTensor = self.create_sentinel_ids(mask_indices_t)
        labels_sentinel_t: ByteTensor = self.create_sentinel_ids(1-mask_indices_t)

        input_ids_t: ShortTensor = torch.from_numpy(input_ids).to(self.device)
        batch_input_ids: ShortTensor = self.filter_input_ids(input_ids_t, input_ids_sentinel_t, result_pad=self.pad_token_id)
        batch_labels: ShortTensor = self.filter_input_ids(input_ids_t, labels_sentinel_t, result_pad=self.label_ignore_index)

        attention_mask: BoolTensor = batch_input_ids != self.pad_token_id
        decoder_attention_mask: BoolTensor = batch_labels != self.label_ignore_index

        # verify that batch_input_ids ends with </s> followed by padding √
        # verify that attention mask reveals </s> √
        # verify that batch_labels ends with </s> followed by padding √
        # verify that decoder_attention_mask reveals </s> √
        data = BooruBatchData(
            input_ids=batch_input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            labels=batch_labels.detach().cpu(),
            decoder_attention_mask=decoder_attention_mask.detach().cpu(),
        )
        batch_encoding = BatchEncoding(
            data=data,
        )

        return batch_encoding

    def create_sentinel_ids(self, mask_indices: ByteTensor) -> ByteTensor:
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices: ByteTensor = mask_indices - mask_indices.roll(1, dims=-1) & mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids: ByteTensor = start_indices.cumsum(-1, dtype=torch.int8).where(start_indices != 0, start_indices)
        sentinel_ids = (self.sentinel_start_ix + sentinel_ids).where(sentinel_ids != 0, 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids: ShortTensor, sentinel_ids: ByteTensor, result_pad: int) -> ShortTensor:
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length.
        """
        input_ids_full: ByteTensor = sentinel_ids.where(sentinel_ids.bool(), input_ids)
        longest_after_masking: int = (input_ids_full > self.pad_token_id).sum(-1).max().item()
        desired_length: int = longest_after_masking+1
        if self.pad_to_multiple is not None:
            desired_length = ceil_to_multiple(desired_length, self.pad_to_multiple)
        if self.max_length is not None:
            assert desired_length <= self.max_length
        retaineds: ByteTensor = full(
            (input_ids_full.size(0), desired_length),
            fill_value=result_pad,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        for out_row, in_row, retain in zip(retaineds, input_ids_full, input_ids_full > self.pad_token_id):
            retained: ByteTensor = in_row.masked_select(retain)
            out_row[:retained.size(-1)] = retained
            out_row[retained.size(-1)] = self.eos_token_id

        return retaineds