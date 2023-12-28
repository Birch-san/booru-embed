from os import listdir
from os.path import dirname, realpath, join, abspath
from pathlib import Path
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal
import numpy as np
from numpy.typing import NDArray
from functools import partial
import torch
from torch import LongTensor
from transformers import HfArgumentParser
from datasets import DatasetDict

from src.vocab import Vocab
from src.booru_dataset import BooruDataset, BucketContent, RandomSpansNoiseMask
from src.random_spans_noise_mask import random_spans_noise_mask
from src.nnlshp import pack_using_nnlshp

@dataclass
class DataArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    collator_device: Literal['cpu', 'cuda'] = field(default='cpu', metadata={"help": "Device used for vectorizable operations in BooruCollator. My gut feeling is that there is enough data that CUDA should help, but in practice I think CPU is benchmarking better, especially for dataloader_num_workers>1. Maybe using CUDA whilst the model is running causes contention.", "choices": ["cpu", "cuda"]})
    pad_to_multiple: Optional[int] = field(default=None, metadata={"help": "Collator can pad sequence lengths to a multiple. Multiples such as 8 or 64 are required to utilize tensor cores. Multiples of 8 are required to support attention bias, if --xformers is enabled."})

def main():
    parser = HfArgumentParser((DataArgs,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, *_ = parser.parse_json_file(json_file=abspath(sys.argv[1]))
    else:
        data_args, *_ = parser.parse_args_into_dataclasses()

    # TODO: make a BooruTokenizer class, with an interface a bit more typical of transformers Tokenizers
    vocab = Vocab()
    # create this file by running scripts/make_tokenizer.py
    with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
        vocab.load(vocab_in)
    assert len(vocab.tokens) < (1<<15), "we load our tokenized dataset in int16, which assumes a tokenizer's vocab being smaller than a signed 16-bit integer."

    device = torch.device('cuda')

    script_dir = Path(dirname(realpath(__file__)))
    repo_root: Path = script_dir.parent
    # populate this folder by running tsv_bucketer.py
    # we used to group prompts into buckets by nearest length,
    # but since it seems hard to make HF dataloader source batch content from alike buckets:
    # we've switched to just "everything in the same bucket, <=255 length"
    # in_dir = repo_root.joinpath('out_lenbucket')
    in_dir = repo_root.joinpath('out_onebucket')

    potential_bucket_dirs: List[str] = listdir(in_dir)
    bucket_values: List[int] = [int(dir.lstrip('b')) for dir in potential_bucket_dirs if bool(re.fullmatch(r'b[0-9]+', dir))]
    bucket_values.sort()
    bucket_dirs: List[str] = [join(in_dir, f'b{val}') for val in bucket_values]

    bucket_samples_train: Dict[int, BucketContent] = {}
    bucket_samples_test: Dict[int, BucketContent] = {}

    def get_indices(lengths: NDArray) -> NDArray:
        indices: LongTensor = torch.from_numpy(lengths).to(device=device, dtype=torch.long)
        indices.cumsum_(0)
        indices.resize_(indices.shape[0]+1)
        indices = indices.roll(1)
        return indices.cpu().numpy()

    train_test_split = data_args.validation_split_percentage/100
    # for testing
    sample_limit_per_bucket: Optional[int] = None if data_args.max_train_samples is None else (
        int(data_args.max_train_samples/(1-train_test_split))
    )
    # let's add BOS to each of these
    added_tokens = 1
    for bucket_value, bucket_dir in zip(bucket_values, bucket_dirs):
        values: NDArray = np.load(join(bucket_dir, 'values.npy'))
        lengths: NDArray = np.load(join(bucket_dir, 'lengths.npy'))
        lens: LongTensor = torch.as_tensor(lengths, device=device, dtype=torch.long) + added_tokens
        histogram: LongTensor = torch.histc(lens, bins=256, min=added_tokens, max=255 + added_tokens)
        # max_sequences_per_pack=4 worked fine, =8 exceeded my RAM
        samples_to_take = lengths.shape[0] if sample_limit_per_bucket is None else min(lengths.shape[0], sample_limit_per_bucket)

        lengths = lengths[:samples_to_take]
        train_start_ix = int(samples_to_take * train_test_split)

        test_indices: NDArray = get_indices(lengths[:train_start_ix])
        train_indices: NDArray = get_indices(lengths[train_start_ix:samples_to_take])
        del lengths

        # TODO:
        # pull the new BOS tokenizer and dataset
        # consider each strategy
        #   [[1, 255], [2, 254], [3, 253], ..]
        # and each strategy count
        #   np.array([blah, blah, blah])
        # create a BooruDataset-like ragged-array accessor?
        # create a Dict[int, List[int]]
        #   bucket_len -> indices at which sequences with such lengths can be found
        # hmm so basically group indices by lengths? not that there'll be any way to do that..
        # allocate a [strategy_repeat_count.cumsum(), data_args.max_seq_length] buffer, inited with PAD token
        # allocate a [strategy_repeat_count.cumsum(), max_sequences_per_pack] buffer, inited with 0
        #   or maybe we just want a strategy index, [strategy_repeat_count.cumsum(), 1]
        # then iterate through each zip(strategy, repeat_count)
        # within that, iterate through each range(repeat_count)
        # lookup sequences of the lengths you need, from the dict
        # build samples by concatenating each such sequence. having padding left at the end is fine
        # assign each such sample into our buffer
        # and write into max_sequences_per_pack the lengths from the strategy?
        # or maybe we just want a strategy index? and knowledge of what max_sequences_per_pack and max_sequence_length were set to
        strategy_set, strategy_repeat_count = pack_using_nnlshp(histogram.cpu().numpy(), max_sequence_length=data_args.max_seq_length, max_sequences_per_pack=2)

        bucket_samples_test[bucket_value] = BucketContent(
            values = values[:test_indices[-1]],
            indices = test_indices,
        )
        bucket_samples_train[bucket_value] = BucketContent(
            values = values[test_indices[-1]:test_indices[-1]+train_indices[-1]],
            indices = train_indices,
        )
        del values, test_indices, train_indices
        break # just peeking in first bucket for now (note: nowadays there's only one bucket anyway)

    # [[vocab.tokens[token_ix] for token_ix in bucket_samples_test[bucket_value].values[start:end]] for start, end in pairwise(bucket_samples_test[bucket_value].indices)]
    # [[vocab.tokens[token_ix] for token_ix in bucket_samples_train[bucket_value].values[start:end]] for start, end in pairwise(bucket_samples_train[bucket_value].indices)]

    random_spans_noise_mask_: RandomSpansNoiseMask = partial(
        random_spans_noise_mask,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )
    train_dataset = BooruDataset(
        bucket_content=bucket_samples_train[bucket_value],
        random_spans_noise_mask=random_spans_noise_mask_,
        # vocab is optional, to aid in debugging (enables decoding of a sample)
        vocab=vocab,
    )
    test_dataset = BooruDataset(
        bucket_content=bucket_samples_test[bucket_value],
        random_spans_noise_mask=random_spans_noise_mask_,
        # vocab is optional, to aid in debugging (enables decoding of a sample)
        vocab=vocab,
    )
    del bucket_samples_train, bucket_samples_test
    tokenized_datasets = DatasetDict({
        'train': train_dataset,
        'validation': test_dataset,
    })

    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

if __name__ == "__main__":
    main()