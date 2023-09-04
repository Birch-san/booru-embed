from dataclasses import dataclass, field
from transformers import BatchEncoding
from typing import List, Optional, TypedDict
import numpy as np
from numpy.typing import NDArray
import torch
from torch import ByteTensor, BoolTensor, ShortTensor, full
from torch.nn.functional import pad
from transformers.utils.import_utils import _is_package_available

from .booru_dataset import BooruDatum
from .vocab import Vocab

class BooruBatchData(TypedDict):
    input_ids: ShortTensor
    attention_mask: BoolTensor
    labels: ShortTensor
    decoder_attention_mask: BoolTensor

@dataclass
class BooruDataCollatorForT5MLM:
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
    decoder_start_token_id: int
    # for debug (enables decoding of captions)
    vocab: Optional[Vocab] = None
    device: torch.device = field(init=False)
    # xformers kernels only support attention bias for sequence lengths multiple of 8
    pad_to_multiple: Optional[int] = None

    def __post_init__(self):
        self.device = torch.device('cuda')
        assert self.pad_token_id == 0, 'we currently employ in filter_input_ids a condition which ignores both -1 tokens and pad tokens via the criteria `<= 0` (on the basis it may be cheaper than `!= -1 & != self.pad_token_id`). to use a non-zero pad_token_id: we would need to remove this optimization.'

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
        input_ids_sentinel_t: ByteTensor = self.create_sentinel_ids(mask_indices_t)
        labels_sentinel_t: ByteTensor = self.create_sentinel_ids(1-mask_indices_t)

        # TODO: maybe insertion of conv and EOS tokens should be job of collator
        # so that they can't be masked-out
        input_ids_t: ShortTensor = torch.from_numpy(input_ids).to(self.device)
        # TODO: these are still on-GPU, and this is multiprocess. does that leak? can they be sent inter-process? is a queue required?
        batch_input_ids: ShortTensor = self.filter_input_ids(input_ids_t, input_ids_sentinel_t)
        batch_labels: ShortTensor = self.filter_input_ids(input_ids_t, labels_sentinel_t)

        # TODO: should we introduce conv tokens inside the model itself? (encoder could do it)

        attention_mask: BoolTensor = batch_input_ids != self.pad_token_id
        decoder_attention_mask: BoolTensor = batch_labels != self.pad_token_id

        if self.pad_to_multiple is not None:
            input_length = batch_input_ids.shape[-1]
            input_extra_tokens_needed = self.pad_to_multiple - (input_length % self.pad_to_multiple)
            # pad to multiple of (for example, 8) tokens
            batch_input_ids = pad(batch_input_ids, pad=(0, input_extra_tokens_needed), value=self.pad_token_id)
            attention_mask = pad(attention_mask, pad=(0, input_extra_tokens_needed))

            # TODO: do labels need padding too?
            # TODO: is it actually (input_ids + labels) we need to pad, rather than the sequences individually?
            label_length = batch_labels.shape[-1]
            label_extra_tokens_needed = self.pad_to_multiple - (label_length % self.pad_to_multiple)
            batch_labels = pad(batch_labels, pad=(0, label_extra_tokens_needed), value=self.pad_token_id)
            decoder_attention_mask = pad(decoder_attention_mask, pad=(0, label_extra_tokens_needed))

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

    def filter_input_ids(self, input_ids: ShortTensor, sentinel_ids: ByteTensor) -> ShortTensor:
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length.
        """
        input_ids_full: ByteTensor = sentinel_ids.where(sentinel_ids.bool(), input_ids)
        longest_after_masking: int = (input_ids_full > self.pad_token_id).sum(-1).max().item()
        retaineds: ByteTensor = full(
            (input_ids_full.size(0), longest_after_masking+1),
            fill_value=self.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        for out_row, in_row, retain in zip(retaineds, input_ids_full, input_ids_full > self.pad_token_id):
            retained: ByteTensor = in_row.masked_select(retain)
            out_row[:retained.size(-1)] = retained
            out_row[retained.size(-1)] = self.eos_token_id

        return retaineds