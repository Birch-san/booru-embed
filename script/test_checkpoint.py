import torch
from dataclasses import dataclass, field
from typing import Optional, Literal
from logging import getLogger
from transformers.utils.import_utils import _is_package_available
from transformers import CONFIG_MAPPING, HfArgumentParser
import sys
import os

from src.model.modeling_t5_booru import T5BooruForMaskedLM
from src.model.configuration_t5_booru import T5BooruConfig
from src.vocab import Vocab
from src.booru_special_tokens import SpecialToken, make_mask_token, make_vocab_pad_token
from src.ceil_to_multiple import remaining_to_multiple

logger = getLogger(__name__)

_xformers_available: bool = _is_package_available('xformers')

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
    parser = HfArgumentParser((ModelArguments, SysArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, sys_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, sys_args = parser.parse_args_into_dataclasses()

    device = torch.device('cuda')

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
        variant='gamma_g',
    )
    
    if model_args.xformers:
        assert _xformers_available, 'You requested xformers, but the xformers package does not appear to be installed.'
        assert torch.cuda.is_available(), "You requested xformers, but CUDA is not available (you would not be able to use xformers' accelerated CUDA kernels)."
        model.enable_xformers_memory_efficient_attention()
    else:
        if _xformers_available and torch.cuda.is_available():
            logger.warning('xformers is available, but you are not using it.')
    
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

if __name__ == "__main__":
    main()