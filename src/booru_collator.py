from dataclasses import dataclass
from transformers import BatchEncoding
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray

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

    def __call__(self, examples: List[BooruDatum]) -> BatchEncoding:
        # tensorize input
        input_ids: NDArray = np.vstack([e[0] for e in examples])
        lengths: NDArray = np.hstack([e[1] for e in examples], dtype=np.int16)
        mask_indices: NDArray = np.vstack([e[2] for e in examples], dtype=np.int16)

        # TODO: we could use lengths.max() to make batch smaller in edge-case where every batch item is smaller than bucket max length
        max_len: int = input_ids.shape[-1]

        # TODO: random_spans_noise_mask doesn't seem to work for lengths 30 and below
        mask_indices = np.vstack([self.random_spans_noise_mask(length, pad_to=max_len) for length in lengths])
        labels_mask = ~mask_indices

        input_ids_sentinel: NDArray = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel: NDArray = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

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