import torch
from torch import LongTensor, inference_mode
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal, Iterable
import logging
from logging import getLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.import_utils import _is_package_available
from transformers.utils.versions import require_version
from transformers import HfArgumentParser, TrainingArguments, GenerationConfig, set_seed
from transformers.generation.streamers import BaseStreamer
import sys
import os
from os import listdir
from os.path import dirname, realpath, join
from datasets import DatasetDict
from functools import partial
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import re

from src.model.modeling_t5_booru import T5BooruForMaskedLM
from src.model.configuration_t5_booru import T5BooruConfig
from src.vocab import Vocab
from src.booru_special_tokens import SpecialToken, make_mask_token, make_vocab_pad_token
from src.booru_collator_t5_mlm import BooruDataCollatorForT5MLM
from src.booru_dataset import BooruDataset, BucketContent, RandomSpansNoiseMask
from src.random_spans_noise_mask import random_spans_noise_mask
from src.ceil_to_multiple import remaining_to_multiple
from src.booru_collator import BooruBatchData

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = getLogger(__name__)

_xformers_available: bool = _is_package_available('xformers')

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
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

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    xformers: bool = field(
        default=False,
        metadata={
            "help": (
                'Whether to use xformers memory_efficient_attention instead of the default torch sdp attention.'
                'xformers has accelerated kernels for attention bias, whereas torch sdp does not appear to currently.'
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

@dataclass
class SysArguments:
    allow_bf16_reduced_precision_reduction: Optional[bool] = field(default=None, metadata={"help": 'torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction; https://pytorch.org/docs/stable/notes/cuda.html'})
    allow_fp16_reduced_precision_reduction: Optional[bool] = field(default=None, metadata={"help": 'torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction; https://pytorch.org/docs/stable/notes/cuda.html'})
    allow_tf32: Optional[bool] = field(default=None, metadata={"help": 'torch.backends.cuda.matmul.allow_tf32; https://pytorch.org/docs/stable/notes/cuda.html'})
    cudnn_allow_tf32: Optional[bool] = field(default=None, metadata={"help": 'torch.backends.cudnn.allow_tf32; https://pytorch.org/docs/stable/notes/cuda.html'})
    float32_matmul_precision: Optional[Literal['highest', 'high', 'medium']] = field(default=None, metadata={"help": 'torch.set_float32_matmul_precision(); https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html', "choices": ['highest', 'high', 'medium']})

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, SysArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, sys_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, sys_args = parser.parse_args_into_dataclasses()

    device = torch.device('cuda')

    if data_args.collator_device == 'cuda':
        # I think this may only be necessary because by default we are using pinning. but pinning sounds like a good thing.
        logger.info("Setting torch.multiprocessing start_method to 'spawn', because data collators will be using CUDA.")
        torch.multiprocessing.set_start_method('spawn')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config: T5BooruConfig = T5BooruConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config: T5BooruConfig = T5BooruConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # TODO: make a BooruTokenizer class, with an interface a bit more typical of transformers Tokenizers
    vocab = Vocab()
    # create this file by running scripts/make_tokenizer.py
    with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
        vocab.load(vocab_in)
    assert config.vocab_size_nominal == len(vocab.tokens), f"config.vocab_size_nominal != len(vocab.tokens) ({config.vocab_size_nominal} != {len(vocab.tokens)}). we will construct model's Embedding from config, and we will want all the tokenizer's tokens represented in the Embedding."
    if config.pad_vocab_to_multiple:
        for ix in range(remaining_to_multiple(len(vocab.tokens), config.pad_vocab_to_multiple)):
            vocab.add_token(make_vocab_pad_token(ix))
    assert config.vocab_size == len(vocab.tokens), f"config.vocab_size != len(vocab.tokens) ({config.vocab_size} != {len(vocab.tokens)}). after padding our Vocab to multiple of config.pad_vocab_to_multiple={config.pad_vocab_to_multiple}: we did not reach the config.vocab_size={config.vocab_size}, but rather {len(vocab.tokens)}."
    assert config.vocab_size % config.pad_vocab_to_multiple == 0, f"something has gone wrong with the maths, and our vocab did not actually end up as a multiple of {config.pad_vocab_to_multiple}, after padding it."
    assert len(vocab.tokens) < (1<<15), "we load our tokenized dataset in int16, which assumes a tokenizer's vocab being smaller than a signed 16-bit integer."

    model: T5BooruForMaskedLM = T5BooruForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        # variant='gamma_g',
    )
    
    if model_args.xformers:
        assert _xformers_available, 'You requested xformers, but the xformers package does not appear to be installed.'
        assert torch.cuda.is_available(), "You requested xformers, but CUDA is not available (you would not be able to use xformers' accelerated CUDA kernels)."
        model.enable_xformers_memory_efficient_attention()
    elif _xformers_available and torch.cuda.is_available():
        logger.warning('xformers is available, but you are not using it.')
    
    # for debug
    model.vocab = vocab

    embedding_size = model.get_input_embeddings().weight.shape[0]
    assert embedding_size == len(vocab.tokens)

    max_seq_length: int = min(data_args.max_seq_length or config.max_ctx_len, config.max_ctx_len)
    
    # https://pytorch.org/docs/stable/notes/cuda.html
    if sys_args.allow_bf16_reduced_precision_reduction is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = sys_args.allow_bf16_reduced_precision_reduction
    if sys_args.allow_fp16_reduced_precision_reduction is not None:    
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = sys_args.allow_fp16_reduced_precision_reduction
    if sys_args.allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = sys_args.allow_tf32
    if sys_args.cudnn_allow_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = sys_args.cudnn_allow_tf32
    if sys_args.float32_matmul_precision is not None:
        # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
        torch.set_float32_matmul_precision(sys_args.float32_matmul_precision)

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
    for bucket_value, bucket_dir in zip(bucket_values, bucket_dirs):
        values: NDArray = np.load(join(bucket_dir, 'values.npy'))
        lengths: NDArray = np.load(join(bucket_dir, 'lengths.npy'))
        samples_to_take = lengths.shape[0] if sample_limit_per_bucket is None else min(lengths.shape[0], sample_limit_per_bucket)

        lengths = lengths[:samples_to_take]
        train_start_ix = int(samples_to_take * train_test_split)

        test_indices: NDArray = get_indices(lengths[:train_start_ix])
        train_indices: NDArray = get_indices(lengths[train_start_ix:samples_to_take])
        del lengths

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

    pad_to_multiple: Optional[int] = data_args.pad_to_multiple
    if model_args.xformers:
        assert pad_to_multiple is not None and pad_to_multiple % 8 == 0, "To enable xformers: you must ensure that --pad_to_multiple is set to a multiple of 8."

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = BooruDataCollatorForT5MLM(
        eos_token_id=vocab.token_to_ix[SpecialToken.EOS.value],
        pad_token_id=vocab.token_to_ix[SpecialToken.Pad.value],
        label_ignore_index=config.label_ignore_index,
        sentinel_start_ix=vocab.token_to_ix[make_mask_token(0)],
        decoder_start_token_id=config.decoder_start_token_id,
        # vocab is optional, to aid in debugging (enables decoding of a sample)
        vocab=vocab,
        # xformers kernels only support attention bias for sequence lengths multiple of 8
        pad_to_multiple=pad_to_multiple,
        pad_to_max=data_args.pad_to_max_length,
        max_length=max_seq_length,
        device=torch.device(data_args.collator_device),
        include_unmasked=True,
    )
    
    model.to(device)

    @dataclass
    class Streamer(BaseStreamer):
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
    batch_size=training_args.per_device_eval_batch_size
    streamer=Streamer(vocab=vocab, batch_size=batch_size)

    data_loader = DataLoader(
        tokenized_datasets['validation'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_args.preprocessing_num_workers or 0,
        collate_fn=data_collator,
    )
    batches: Iterable[BooruBatchData] = data_loader
    for batch in batches:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)
        unmasked = batch['unmasked'].to(device)
        # [[vocab.tokens[token_ix] for token_ix in caption] for caption in batch['input_ids']]
        # print('\n'.join(''.join('1' if tok else '0' for tok in mask) for mask in batch['attention_mask'].byte()))
        # [[-100 if token_ix == -100 else vocab.tokens[token_ix] for token_ix in caption] for caption in batch['labels']]
        # [[vocab.tokens[token_ix] for token_ix in caption] for caption in batch['decoder_input_ids']]
        # print('\n'.join(''.join('1' if tok else '0' for tok in mask) for mask in batch['decoder_attention_mask'].byte()))
        # [[vocab.tokens[token_ix] for token_ix in caption] for caption in batch['unmasked']]
        max_new_tokens = 20
        out = model.generate(
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens,
            ),
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
        )
        # out.acc_tok
        # out.decoded

        ####
        #### and now we try to do the same greedy search as generate, manually (no currently working):
        ####
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            inputs=None, bos_token_id=None, model_kwargs=model_kwargs,
        )
        model_kwargs['use_cache']=True

        # encoder_outputs is not in model_kwargs, so we do this:
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        acc_tok: LongTensor = torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor: LongTensor = torch.tensor([vocab.token_to_ix[SpecialToken.EOS.value]], device=input_ids.device)

        for _ in range(max_new_tokens):
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            with inference_mode():
                outputs = model.forward(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            encoder_last_hidden_state = outputs.encoder_last_hidden_state

            next_token_logits = outputs.logits[:, -1, :]

            next_tokens = torch.argmax(next_token_logits, dim=-1)
            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + vocab.token_to_ix[SpecialToken.Pad.value] * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            acc_tok = torch.cat([
                acc_tok,
                next_tokens.unsqueeze(-1),
            ], dim=-1)

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=True
            )

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    break
        # [[vocab.tokens[token_ix] for token_ix in caption] for caption in acc_tok]
        pass
    pass

if __name__ == "__main__":
    main()