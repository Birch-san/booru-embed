from dataclasses import dataclass, field
from transformers import BatchEncoding
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
import torch
from torch import ByteTensor, BoolTensor, LongTensor, arange
from torch.nn.functional import pad

from .booru_dataset import BooruDatum

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
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    eos_token_id: int
    sentinel_start_ix: int
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.device = torch.device('cuda')

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

        # TODO: random_spans_noise_mask doesn't seem to work for lengths 30 and below
        labels_mask = ~mask_indices

        input_ids_sentinel: NDArray = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel: NDArray = self.create_sentinel_ids(labels_mask.astype(np.int8))

        mask_indices_t: BoolTensor = torch.from_numpy(mask_indices).to(self.device, torch.int8)
        input_ids_sentinel_t: ByteTensor = self.create_sentinel_ids_torch(mask_indices_t)
        labels_sentinel_t: ByteTensor = self.create_sentinel_ids_torch(1-mask_indices_t)

        # numpy imply doesn't support batch of varied-length samples
        # it seems to somehow obliterate our final conv token and EOS (or maybe that got masked-out)
        # and puts an EOS in wrong place (end of padding)
        batch_input_ids: NDArray = self.filter_input_ids(input_ids[:1], input_ids_sentinel[:1])
        batch_labels: NDArray = self.filter_input_ids(input_ids[:1], labels_sentinel[:1])

        # TODO: maybe insertion of conv and EOS tokens should be job of collator
        # so that they can't be masked-out
        input_ids_t: LongTensor = torch.from_numpy(input_ids).to(self.device)
        batch_input_ids_t: ByteTensor = self.filter_input_ids_torch(input_ids_t, input_ids_sentinel_t)
        batch_labels_t: ByteTensor = self.filter_input_ids_torch(input_ids_t, labels_sentinel_t)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        # TODO: model supports doing this internally, so we could get rid of this
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        # TODO: model is receiving ndarray rather than tensor
        return batch

    def create_sentinel_ids_torch(self, mask_indices: ByteTensor) -> ByteTensor:
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

    def create_sentinel_ids(self, mask_indices: NDArray) -> NDArray:
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, self.sentinel_start_ix + sentinel_ids, 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids_torch(self, input_ids: LongTensor, sentinel_ids: ByteTensor) -> LongTensor:
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        # batch_size: int = input_ids.shape[0]

        input_ids_full: ByteTensor = sentinel_ids.where(sentinel_ids != 0, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids: ByteTensor = input_ids_full.gather(-1, arange(input_ids_full.shape[-1], device=self.device).where(input_ids_full >= 0, 0))
        # input_ids_eos: ByteTensor = pad(input_ids, pad=(0, 1), mode='constant', value=self.pad_token_id)
        # input_ids_eos.argmin(-1) tells you where to insert eos_token,
        # but actually it's the same as lengths
        # and actually we already inserted eos token
        # input_ids = np.concatenate(
        #     [input_ids, np.full((batch_size, 1), self.eos_token_id, dtype=np.int32)], axis=-1
        # )
        return input_ids

    def filter_input_ids(self, input_ids: NDArray, sentinel_ids: NDArray) -> NDArray:
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids