#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
from os import listdir
from os.path import dirname, realpath, join
from pathlib import Path
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
from numpy.typing import NDArray
from functools import partial
import torch
from torch import LongTensor
from itertools import pairwise

import datasets
import evaluate
from datasets import DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    T5Config,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from src.vocab import Vocab
from src.model.modeling_t5_booru import T5BooruForMaskedLM
from src.compute_input_and_target_lengths import compute_input_and_target_lengths
from src.booru_special_tokens import SpecialToken, make_mask_token
from src.booru_collator import BooruDataCollatorForT5MLM
from src.booru_dataset import BooruDataset, BucketContent, RandomSpansNoiseMask
from src.random_spans_noise_mask import random_spans_noise_mask

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.32.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
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
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

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

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

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
        config: T5Config = T5Config.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config: T5Config = T5Config.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # TODO: make a BooruTokenizer class, with an interface a bit more typical of transformers Tokenizers
    vocab = Vocab()
    # create this file by running scripts/make_tokenizer.py
    with open('out_tokenizer/vocab.txt', mode='r', encoding='utf-8') as vocab_in:
        vocab.load(vocab_in)
    assert len(vocab.tokens) < (1<<15), "we load our tokenized dataset in int16, which assumes a tokenizer's vocab being smaller than a signed 16-bit integer."

    if model_args.model_name_or_path:
        model: T5BooruForMaskedLM = T5BooruForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model: T5BooruForMaskedLM = T5BooruForMaskedLM(config)
    
    # for debug
    model.vocab = vocab

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(vocab.tokens) > embedding_size:
        model.resize_token_embeddings(len(vocab.tokens))

    # TODO: save this into the model config or something
    model_max_ctx_len = 255

    max_seq_length: int = min(data_args.max_seq_length or model_max_ctx_len, model_max_ctx_len)

    if data_args.max_seq_length is None:
        max_seq_length = model_max_ctx_len
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `model_max_ctx_len` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > model_max_ctx_len:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({model_max_ctx_len}). Using max_seq_length={model_max_ctx_len}."
            )
        max_seq_length = min(data_args.max_seq_length, model_max_ctx_len)
    
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

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = BooruDataCollatorForT5MLM(
        eos_token_id=vocab.token_to_ix[SpecialToken.EOS.value],
        pad_token_id=vocab.token_to_ix[SpecialToken.Pad.value],
        sentinel_start_ix=vocab.token_to_ix[make_mask_token(0)],
        decoder_start_token_id=model.config.decoder_start_token_id,
        # vocab is optional, to aid in debugging (enables decoding of a sample)
        vocab=vocab,
    )

    # Initialize our Trainer
    # trainer = BooruTrainer(
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # tokenizer=tokenizer,
        # tokenizer=tokenize_function,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
