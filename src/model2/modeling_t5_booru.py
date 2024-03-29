# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model."""


import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union, Callable, NamedTuple, Sequence, List

import torch
from torch import nn, FloatTensor, LongTensor, ShortTensor
from torch.nn import CrossEntropyLoss, Conv1d, Embedding, Linear, Parameter
from torch.nn.functional import scaled_dot_product_attention, pad, conv1d
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from functools import reduce

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import _is_package_available
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
# from transformers.models.t5.configuration_t5 import T5Config
from .configuration_t5_booru import T5BooruConfig, SReparamConfig
from .sigma_reparam import SReparam, SReparamSupportedModule

from ..vocab import Vocab
from ..z_loss import ZLoss
from ..ceil_to_multiple import remaining_to_multiple, ceil_to_multiple
from .compile_wrap import compile_wrap


logger = logging.get_logger(__name__)

_xformers_available: bool = _is_package_available('xformers')

if _xformers_available:
    import xformers
    import xformers.ops
else:
    xformers = None

XFORMERS_NEG_BIAS=-10000

_CONFIG_FOR_DOC = "T5BooruConfig"
_CHECKPOINT_FOR_DOC = "t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5BooruForMaskedLM.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5BooruForMaskedLM.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


class T5BooruLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


@compile_wrap
def rms_norm(x: FloatTensor, scale: FloatTensor, eps: float) -> FloatTensor:
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, shape: Sequence[int], eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x: FloatTensor) -> FloatTensor:
        return rms_norm(x, self.scale, self.eps)

try:
    from apex.normalization import FusedRMSNorm

    T5BooruLayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of T5BooruLayerNorm")
except ImportError:
    # using the normal T5BooruLayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to T5BooruLayerNorm")
    pass

# T5BooruLayerNorm = RMSNorm

ALL_LAYERNORM_LAYERS.append(T5BooruLayerNorm)

def unwrap_sreparam(mod: SReparamSupportedModule | SReparam[SReparamSupportedModule]) -> SReparamSupportedModule:
    return mod.op if isinstance(mod, SReparam) else mod

class KeyValue(NamedTuple):
    key: FloatTensor
    value: FloatTensor

class AttnOutputs(NamedTuple):
    attn_output: FloatTensor
    kv: Optional[KeyValue]
    position_bias: Optional[FloatTensor]

class AttnOutputsWithWeights(NamedTuple):
    attn_output: FloatTensor
    kv: Optional[KeyValue]
    position_bias: Optional[FloatTensor]
    weights: FloatTensor

class T5BooruDenseActDense(nn.Module):
    def __init__(self, config: T5BooruConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        if config.use_sigma_reparam:
            self.wi = SReparam(self.wi, **config.s_reparam_config)
            self.wo = SReparam(self.wo, **config.s_reparam_config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        wo_weight: FloatTensor = unwrap_sreparam(self.wo).weight
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != wo_weight.dtype
            and wo_weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(wo_weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5BooruDenseGatedActDense(nn.Module):
    def __init__(self, config: T5BooruConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        if config.use_sigma_reparam:
            self.wi_0 = SReparam(self.wi_0, **config.s_reparam_config)
            self.wi_1 = SReparam(self.wi_1, **config.s_reparam_config)
            self.wo = SReparam(self.wo, **config.s_reparam_config)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        wo_weight: FloatTensor = unwrap_sreparam(self.wo).weight
        if (
            isinstance(wo_weight, torch.Tensor)
            and hidden_states.dtype != wo_weight.dtype
            and wo_weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(wo_weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states

@compile_wrap
def linear_geglu(x: FloatTensor, weight: FloatTensor, bias: Optional[FloatTensor]=None) -> FloatTensor:
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)

@compile_wrap
def linear_swiglu(x: FloatTensor, weight: FloatTensor, bias: Optional[FloatTensor]=None) -> FloatTensor:
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)

class LinearGEGLU(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x: FloatTensor) -> FloatTensor:
        return linear_geglu(x, self.weight, self.bias)

class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x: FloatTensor) -> FloatTensor:
        return linear_swiglu(x, self.weight, self.bias)

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

class ActDropDense(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.0, up_proj_ctor: Union[LinearGEGLU, LinearSwiGLU]=LinearSwiGLU, up_bias=False, down_bias=False):
        super().__init__() 
        self.up_proj = up_proj_ctor(d_model, d_ff, bias=up_bias)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(Linear(d_ff, d_model, bias=down_bias))

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x

class T5BooruLayerFF(nn.Module):
    def __init__(self, config: T5BooruConfig):
        super().__init__()
        match config.feed_forward_proj:
            case 'gated-gelu':
                ln_ctor = T5BooruLayerNorm
                self.DenseReluDense = T5BooruDenseGatedActDense(config)
            case 'gelu':
                ln_ctor = T5BooruLayerNorm
                self.DenseReluDense = T5BooruDenseActDense(config)
            case 'geglu' | 'swiglu':
                ln_ctor = T5BooruLayerNorm
                # plays better with torch.compile than apex's FusedRMSNorm
                # ln_ctor = RMSNorm
                up_proj_ctor = LinearGEGLU if config.feed_forward_proj == 'geglu' else LinearSwiGLU
                self.DenseReluDense = ActDropDense(config.d_model, config.d_ff, dropout=config.dropout_rate, up_proj_ctor=up_proj_ctor, up_bias=False, down_bias=False)
            case _:
                raise ValueError(f'Unimplemented feed_forward_proj "{config.feed_forward_proj}".')

        self.layer_norm = ln_ctor(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: FloatTensor) -> FloatTensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5BooruAttention(nn.Module):
    use_xformers_attn: bool
    xformers_attention_op: Optional[Callable]
    tied_avg_key_len: Optional[FloatTensor]

    def __init__(
        self,
        config: T5BooruConfig,
        has_relative_attention_bias=False,
        tied_avg_key_len: Optional[FloatTensor] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.scale = config.d_kv**-.5
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.tied_avg_key_len=tied_avg_key_len

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if config.use_sigma_reparam:
            sreparam_heads: Optional[int] = config.num_heads if config.sreparam_multi_head else None
            self.q = SReparam(self.q, heads=sreparam_heads, **config.s_reparam_config)
            self.k = SReparam(self.k, heads=sreparam_heads, **config.s_reparam_config)
            self.v = SReparam(self.v, heads=sreparam_heads, **config.s_reparam_config)
            self.o = SReparam(self.o, **config.s_reparam_config)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

        self.use_xformers_attn = False
        self.xformers_attention_op = None

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        if use_memory_efficient_attention_xformers:
            if not _xformers_available:
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                # Make sure we can run the memory efficient attention
                xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device='cuda'),
                    torch.randn((1, 2, 40), device='cuda'),
                    torch.randn((1, 2, 40), device='cuda'),
                )
                self.use_xformers_attn = True
                self.xformers_attention_op = attention_op
        else:
            self.use_xformers_attn = False
            self.xformers_attention_op = None

    def forward(
        self,
        hidden_states: FloatTensor,
        mask: Optional[FloatTensor] = None,
        input_lengths: Optional[ShortTensor] = None,
        key_value_states: Optional[FloatTensor] = None,
        position_bias: Optional[FloatTensor] = None,
        past_key_value: Optional[Tuple[FloatTensor, FloatTensor]] = None,
        layer_head_mask: Optional[FloatTensor] = None,
        query_length: Optional[int] = None,
        use_cache=False,
        output_attentions=False,
    ) -> AttnOutputs | AttnOutputsWithWeights:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[1 if self.use_xformers_attn else 2] if query_length is None else query_length
            # TODO: check if this length contribution is ever anything other than 1

        key_length: int = real_seq_length if key_value_states is None else key_value_states.shape[1]

        if past_key_value is None or key_value_states is not None:
            extra_key_length_from_past: int = 0
        else:
            # this is all a best-effort/upper-bound, since we don't know what mask was used over the past key
            extra_key_length_from_past: int = past_key_value[0].shape[1 if self.use_xformers_attn else 2]
        
        if extra_key_length_from_past>1:
            pass # friendly place to put breakpoint, to learn about this situation

        def shape(states: FloatTensor) -> FloatTensor:
            """projection"""
            view: FloatTensor = states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
            if not self.use_xformers_attn:
                view = view.transpose(1, 2)
            return view

        def unshape(states: FloatTensor) -> FloatTensor:
            """reshape"""
            if not self.use_xformers_attn:
                states = states.transpose(1, 2).contiguous()
            return states.view(batch_size, -1, self.inner_dim)

        def project(
            hidden_states: FloatTensor,
            proj_layer: nn.Linear,
            key_value_states: Optional[FloatTensor],
            past_key_value: Optional[FloatTensor],
        ) -> FloatTensor:
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # sdp:      (batch_size, n_heads, seq_len, dim_per_head)
                # xformers: (batch_size, seq_len, n_heads, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # sdp:      (batch_size, n_heads, seq_len, dim_per_head)
                # xformers: (batch_size, seq_len, n_heads, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))
            else:
                # inference-only (after prefill; you now have a cache)
                pass

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # sdp:      (batch_size, n_heads, seq_len, dim_per_head)
                    # xformers: (batch_size, seq_len, n_heads, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=1 if self.use_xformers_attn else 2)
                elif past_key_value.shape[1 if self.use_xformers_attn else 2] != key_value_states.shape[1]:
                    # we don't use this code path ourselves
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # sdp:      (batch_size, n_heads, seq_len, dim_per_head)
                    # xformers: (batch_size, seq_len, n_heads, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))
        # sdp:      (batch_size, n_heads, seq_len, dim_per_head)
        # xformers: (batch_size, seq_len, n_heads, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=query_states.device, dtype=query_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=query_states.device)
                # (1, num_heads, query_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
                # (1, num_heads, 1, key_length)

            if mask is not None:
                # mask: (batch, 1, 1, key_length)
                position_bias = position_bias + mask
                # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        if self.tied_avg_key_len is not None and input_lengths is not None:
            # Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis
            # https://arxiv.org/abs/2306.08645
            # instead of scaling logits by:
            #   self.scale
            # we scale by:
            #   self.scale * log(key_len, standard_key_len)**.5
            # where standard_key_len is a "preferred" key length (e.g. median key length in the dataset)
            # this will make the logits more uniform, and thus more entropic, for sequences of different lengths.
            # note: we need it to be at least 2, since log(1) will just make our key disappear
            # also: if we receive a previous key via past_key_value: it gets concatenated to our current key, so we need to account for that
            # caveat: we don't know the mask for the past key. hopefully this only happens for the "decode 1 token at a time" case, where nothing gets masked-out anyway
            key_states *= (extra_key_length_from_past + input_lengths).clamp_min(2).log().div(self.tied_avg_key_len.clamp_min(2).log()).sqrt().to(key_states.dtype).unsqueeze(-1).unsqueeze(-1)

        if self.use_xformers_attn:
            kv_padding = remaining_to_multiple(key_states.size(1), 8)
            if kv_padding:
                # don't reassign key_states and value_states, since these will be output into past_key_values without any masking information
                padded_key_states = pad(key_states, (0, 0, 0, 0, 0, kv_padding))
                padded_value_states = pad(value_states, (0, 0, 0, 0, 0, kv_padding))
                position_bias_masked = pad(position_bias_masked, (0, kv_padding), value=XFORMERS_NEG_BIAS)
            else:
                padded_key_states = key_states
                padded_value_states = value_states
            attn_weights: FloatTensor = xformers.ops.memory_efficient_attention(
                query_states,
                padded_key_states,
                padded_value_states,
                attn_bias=position_bias_masked.to(query_states.dtype).contiguous(),
                p=self.dropout if self.training else 0.,
                op=self.xformers_attention_op,
            ) # [batch, q_len, heads, v_out_dim]
        else:
            attn_weights: FloatTensor = scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=position_bias_masked.to(query_states.dtype),
                dropout_p=self.dropout if self.training else 0.,
            ) # [batch, heads, q_len, v_out_dim]
        
        # assert not attn_weights.isnan().any().item()

        if layer_head_mask is not None:
            # in original implementation, layer_head_mask was compatible with:
            #   [batch, heads, q_len, k_len]
            # due to torch sdp, we are multiplying later, so can only support layer_head_mask broadcastable to:
            #   [batch, heads, q_len, 1]
            if self.use_xformers_attn:
                # in xformers, attn_weights is
                #   [batch, q_len, heads, v_out_dim]
                # so we need to adapt mask from:
                #   [batch, heads, q_len, 1] ->
                #   [batch, q_len, heads, 1]
                layer_head_mask = layer_head_mask.transpose(1, 2)
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(attn_weights)  # (batch_size, seq_length, dim)
        attn_output: FloatTensor = self.o(attn_output)

        # if self.use_xformers_attn:
        #     key_states = key_states.transpose(1,2)
        #     value_states = value_states.transpose(1,2)
        present_key_value_state: Optional[KeyValue] = KeyValue(key_states, value_states) if self.is_decoder and use_cache else None
        outputs = AttnOutputs(attn_output, present_key_value_state, position_bias)

        if output_attentions:
            outputs = AttnOutputsWithWeights(*outputs, attn_weights)
        return outputs


class T5BooruLayerSelfAttention(nn.Module):
    pre_ln: Optional[T5BooruLayerNorm]
    post_ln: Optional[T5BooruLayerNorm]
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        tied_avg_key_len: Optional[FloatTensor] = None,
    ):
        super().__init__()
        self.SelfAttention = T5BooruAttention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            tied_avg_key_len=tied_avg_key_len,
        )
        self.pre_ln = T5BooruLayerNorm(config.d_model, eps=config.layer_norm_epsilon) if config.use_attn_pre_ln else None
        self.post_ln = T5BooruLayerNorm(config.d_model, eps=config.layer_norm_epsilon) if config.use_attn_post_ln else None
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: FloatTensor,
        attention_mask: Optional[FloatTensor] = None,
        input_lengths: Optional[ShortTensor] = None,
        position_bias: Optional[FloatTensor] = None,
        layer_head_mask: Optional[FloatTensor] = None,
        past_key_value: Optional[Tuple[FloatTensor, FloatTensor]] = None,
        use_cache=False,
        output_attentions=False,
    ) -> AttnOutputs | AttnOutputsWithWeights:
        if self.pre_ln is None:
            prenormed_hidden_states = hidden_states
        else:
            prenormed_hidden_states = self.pre_ln(hidden_states)
        attention_outputs: AttnOutputs | AttnOutputsWithWeights = self.SelfAttention(
            prenormed_hidden_states,
            mask=attention_mask,
            input_lengths=input_lengths,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output, *_ = attention_outputs
        layer_output = hidden_states + self.dropout(attn_output)
        if self.post_ln is None:
            postnormed_layer_output = layer_output
        else:
            postnormed_layer_output = self.post_ln(layer_output)
        match attention_outputs:
            case AttnOutputs(_, kv, position_bias):
                return AttnOutputs(postnormed_layer_output, kv, position_bias)
            case AttnOutputsWithWeights(_, kv, position_bias, weights):
                return AttnOutputsWithWeights(postnormed_layer_output, kv, position_bias, weights)
            case _:
                raise ValueError(f"Never heard of attention_outputs type '{type(attention_outputs)}'")


class T5BooruLayerCrossAttention(nn.Module):
    pre_ln: Optional[T5BooruLayerNorm]
    post_ln: Optional[T5BooruLayerNorm]
    def __init__(
        self,
        config,
        tied_avg_key_len: Optional[Parameter] = None,
    ):
        super().__init__()
        self.EncDecAttention = T5BooruAttention(
            config,
            has_relative_attention_bias=False,
            tied_avg_key_len=tied_avg_key_len,
        )
        self.pre_ln = T5BooruLayerNorm(config.d_model, eps=config.layer_norm_epsilon) if config.use_attn_pre_ln else None
        self.post_ln = T5BooruLayerNorm(config.d_model, eps=config.layer_norm_epsilon) if config.use_attn_post_ln else None
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: FloatTensor,
        key_value_states,
        attention_mask: Optional[FloatTensor] = None,
        input_lengths: Optional[ShortTensor] = None,
        position_bias: Optional[FloatTensor] = None,
        layer_head_mask: Optional[FloatTensor] = None,
        past_key_value: Optional[Tuple[FloatTensor, FloatTensor]] = None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ) -> AttnOutputs | AttnOutputsWithWeights:
        if self.pre_ln is None:
            prenormed_hidden_states = hidden_states
        else:
            prenormed_hidden_states = self.pre_ln(hidden_states)
        attention_outputs: AttnOutputs | AttnOutputsWithWeights = self.EncDecAttention(
            prenormed_hidden_states,
            mask=attention_mask,
            input_lengths=input_lengths,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        attn_output, *_ = attention_outputs
        layer_output = hidden_states + self.dropout(attn_output)
        if self.post_ln is None:
            postnormed_layer_output = layer_output
        else:
            postnormed_layer_output = self.post_ln(layer_output)
        match attention_outputs:
            case AttnOutputs(_, kv, position_bias):
                return AttnOutputs(postnormed_layer_output, kv, position_bias)
            case AttnOutputsWithWeights(_, kv, position_bias, weights):
                return AttnOutputsWithWeights(postnormed_layer_output, kv, position_bias, weights)
            case _:
                raise ValueError(f"Never heard of attention_outputs type '{type(attention_outputs)}'")


class T5BooruBlock(nn.Module):
    self_attn: T5BooruLayerSelfAttention
    cross_attn: Optional[T5BooruLayerCrossAttention]
    ffn: Optional[T5BooruLayerFF]
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        tied_ffn: Optional[T5BooruLayerFF] = None,
        tied_self_attn_avg_key_len: Optional[FloatTensor] = None,
        tied_cross_attn_avg_key_len: Optional[FloatTensor] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.self_attn = T5BooruLayerSelfAttention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            tied_avg_key_len=tied_self_attn_avg_key_len,
        )
        if self.is_decoder:
            self.cross_attn = T5BooruLayerCrossAttention(
                config,
                tied_avg_key_len=tied_cross_attn_avg_key_len,
            )

        if config.tie_encoder_ffns and not config.is_decoder:
            assert tied_ffn is not None
            self.ffn = tied_ffn
            # just in case we need a plan B for how to tie FFN,
            # here's a way that's a bit closer to how other people seem to do it.
            # i.e. "construct distinct instances, then make their weights reference-equal" (note: you might tie them later, in the _tie_weights callback).
            # I don't think we need to do this, since we've already made the FFN instances themselves reference-equal.
            # self.ffn = T5BooruLayerFF(config)
            # if config.is_gated_act:
            #     assert isinstance(tied_ffn.DenseReluDense, T5BooruDenseGatedActDense)
            #     assert isinstance(self.ffn.DenseReluDense, T5BooruDenseGatedActDense)
            #     self.ffn.DenseReluDense.wi_0.weight = self.ffn.DenseReluDense.wi_0.weight
            #     self.ffn.DenseReluDense.wi_1.weight = self.ffn.DenseReluDense.wi_1.weight
            # else:
            #     assert isinstance(tied_ffn.DenseReluDense, T5BooruDenseActDense)
            #     assert isinstance(self.ffn.DenseReluDense, T5BooruDenseActDense)
            #     self.ffn.DenseReluDense.wi.weight = self.ffn.DenseReluDense.wi.weight
            # self.ffn.DenseReluDense.wo.weight = self.ffn.DenseReluDense.wo.weight
        else:
            if config.tie_encoder_ffns:
                assert tied_ffn is None, "tying of FFN is not implemented for decoders; we are following 'One Wide FeedForward is All You Need', whose advice for decoders is to remove FFN rather than tie it"
            self.ffn = T5BooruLayerFF(config) if config.decoder_mlp or not config.is_decoder else None

    def forward(
        self,
        hidden_states: FloatTensor,
        attention_mask: Optional[FloatTensor] = None,
        input_lengths: Optional[ShortTensor] = None,
        position_bias=None,
        encoder_hidden_states: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[FloatTensor] = None,
        encoder_input_lengths: Optional[ShortTensor] = None,
        encoder_decoder_position_bias: Optional[FloatTensor] = None,
        layer_head_mask: Optional[FloatTensor] = None,
        cross_attn_layer_head_mask: Optional[FloatTensor] = None,
        past_key_value: None | Tuple[FloatTensor, FloatTensor] | Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor] = None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value: Tuple[FloatTensor, FloatTensor] = past_key_value[:2]
            cross_attn_past_key_value: Tuple[FloatTensor, FloatTensor] = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs: AttnOutputs | AttnOutputsWithWeights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state, *self_bias_and_maybe_weights = self_attention_outputs

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            assert self.cross_attn is not None
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs: AttnOutputs | AttnOutputsWithWeights = self.cross_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                input_lengths=encoder_input_lengths,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, cross_kv_opt, *cross_bias_and_maybe_weights = cross_attention_outputs

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_kv_opt

            # Keep cross-attention outputs and relative position weights
            attention_outputs = tuple(self_bias_and_maybe_weights + cross_bias_and_maybe_weights)
        else:
            attention_outputs = tuple(self_bias_and_maybe_weights)

        # Apply Feed Forward layer
        if self.ffn is not None:
            hidden_states = self.ffn(hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5BooruPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5BooruConfig
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5BooruBlock"]
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5BooruAttention, T5BooruStack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids: LongTensor) -> LongTensor:
        pad_token_id = self.config.pad_token_id
        if self.config.decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
                "See T5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), self.config.decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.detach().roll(1)
            shifted_input_ids[..., 0] = self.config.decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == self.config.label_ignore_index, pad_token_id)

        return shifted_input_ids


class T5BooruStack(T5BooruPreTrainedModel):
    embed_tokens: Optional[Embedding]
    conv_in: Optional[Conv1d|SReparam[Conv1d]]
    use_xformers_attn: bool
    needs_reentrant_checkpoint: bool

    def __init__(
        self,
        config: T5BooruConfig,
        embed_tokens: Optional[Embedding] = None,
        conv_in: Optional[Conv1d|SReparam[Conv1d]] = None,
        tied_ffn: Optional[T5BooruLayerFF] = None,
        tied_self_attn_avg_key_len: Optional[FloatTensor] = None,
        tied_cross_attn_avg_key_len: Optional[FloatTensor] = None,
    ):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.conv_in = conv_in
        self.is_decoder = config.is_decoder
        self.needs_reentrant_checkpoint = config.use_sigma_reparam

        self.block = nn.ModuleList(
            [T5BooruBlock(
                config,
                has_relative_attention_bias=bool(i == 0),
                tied_ffn=tied_ffn,
                tied_self_attn_avg_key_len=tied_self_attn_avg_key_len,
                tied_cross_attn_avg_key_len=tied_cross_attn_avg_key_len,
            ) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5BooruLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        self.use_xformers_attn = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5BooruStack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        if use_memory_efficient_attention_xformers:
            assert _xformers_available, "Memory efficient attention requires xformers to be installed. Run `pip install xformers`."
            self.use_xformers_attn = True
        else:
            self.use_xformers_attn = False

    def forward(
        self,
        input_ids: Optional[torch.IntTensor]=None,
        input_lengths: Optional[torch.ShortTensor]=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_input_lengths: Optional[torch.ShortTensor]=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            # can't embed shorts
            input_ids = input_ids.int()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds: FloatTensor = self.embed_tokens(input_ids)
            if self.conv_in is not None:
                if self.is_decoder:
                    if isinstance(self.conv_in, Conv1d):
                        conv_in: Conv1d = self.conv_in
                    elif isinstance(self.conv_in, SReparam):
                        conv_in: Conv1d = self.conv_in.op
                    else:
                        raise ValueError(f'expected self.conv_in to be Conv1d or SReparam. got "{type(self.conv_in)}"')
                    inputs_embeds = conv1d(
                        inputs_embeds.mT,
                        pad(
                            conv_in.weight[:,:,:-1],
                            pad=(0, 1),
                        ),
                        conv_in.bias,
                        conv_in.stride,
                        conv_in.padding,
                        conv_in.dilation,
                        conv_in.groups,
                    )
                    # TODO: share code with SReparam class
                    if isinstance(self.conv_in, SReparam):
                        if self.conv_in.bias is None:
                            quotient: FloatTensor = (self.conv_in.g / self.conv_in.sigma)
                            inputs_embeds *= quotient.to(inputs_embeds.dtype)
                        else:
                            inputs_embeds = torch.addcmul(
                                self.conv_in.bias.to(inputs_embeds.dtype),
                                inputs_embeds,
                                self.conv_in.g.to(inputs_embeds.dtype),
                                value=(1 / self.conv_in.sigma).to(inputs_embeds.dtype),
                            )
                else:
                    inputs_embeds = self.conv_in(inputs_embeds.mT)
                inputs_embeds = inputs_embeds.mT

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[1 if self.use_xformers_attn else 2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # torch sdp returns NaN if we supply a torch.finfo(torch.float32).min mask cast to bfloat16. but seems fine if it begins in bfloat16.
        mask_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, dtype=mask_dtype)
        if self.use_xformers_attn:
            # xformers returns NaN for maximally-negative biases.
            extended_attention_mask.clamp_min_(XFORMERS_NEG_BIAS)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    input_lengths,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_input_lengths,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_reentrant=self.needs_reentrant_checkpoint,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    input_lengths=input_lengths,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_input_lengths=encoder_input_lengths,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


T5_START_DOCSTRING = r"""

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`T5BooruConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5BooruModel(T5BooruPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    lm_head: Linear|SReparam[Linear]
    shared: Embedding
    decoder_conv_in: Optional[Conv1d|SReparam[Conv1d]]
    encoder_conv_in: Optional[Conv1d|SReparam[Conv1d]]
    tied_ffn: Optional[T5BooruLayerFF]
    
    prompt_avg_key_len: Optional[FloatTensor]
    prompt_avg_key_len_population: Optional[int]
    prompt_avg_key_len_population_max: Optional[int]
    continuation_avg_key_len: Optional[FloatTensor]
    continuation_avg_key_len_population: Optional[int]
    continuation_avg_key_len_population_max: Optional[int]

    wants_prompt_lengths: bool
    wants_continuation_lengths: bool

    def __init__(self, config: T5BooruConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        if config.encoder_conv_in:
            out_channels = config.d_model
            assert config.pad_token_id == 0, "we make use of conv1d with 'zeros' padding_mode, under the assumption that pad token will be 0. if you wish to pad with another token, then you will need to set the Conv1d's padding to none, and pad the sequence yourself with the pad token of your choosing."
            self.encoder_conv_in = Conv1d(in_channels=config.d_model, out_channels=out_channels, kernel_size=3, padding=1, bias=not config.use_sigma_reparam)
            if config.use_sigma_reparam:
                self.encoder_conv_in = SReparam(self.encoder_conv_in, **config.s_reparam_config, bias_shape=(out_channels, 1), v_shape=(1, config.d_model, 1))

            indirection = 'op.' if config.use_sigma_reparam else ''
            self._tied_weights_keys.extend([
                f'encoder.tied_conv_in.{indirection}weight',
                f'encoder.tied_conv_in.bias'
            ])
        else:
            self.encoder_conv_in = None
        
        if config.decoder_conv_in:
            out_channels = config.d_model
            assert config.pad_token_id == 0, "we make use of conv1d with 'zeros' padding_mode, under the assumption that pad token will be 0. if you wish to pad with another token, then you will need to set the Conv1d's padding to none, and pad the sequence yourself with the pad token of your choosing."
            if config.encoder_conv_in and config.tie_conv_in:
                self.decoder_conv_in = self.encoder_conv_in
            else:
                self.decoder_conv_in = Conv1d(in_channels=config.d_model, out_channels=out_channels, kernel_size=3, padding=1, bias=not config.use_sigma_reparam)
            if config.use_sigma_reparam:
                self.decoder_conv_in = SReparam(self.decoder_conv_in, **config.s_reparam_config, bias_shape=(out_channels, 1), v_shape=(1, config.d_model, 1))

            indirection = 'op.' if config.use_sigma_reparam else ''
            self._tied_weights_keys.extend([
                f'decoder.tied_conv_in.{indirection}weight',
                f'decoder.tied_conv_in.bias'
            ])
        else:
            self.decoder_conv_in = None

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        if encoder_config.tie_encoder_ffns:
            for ix in range(encoder_config.num_layers):
                indirection = 'op.' if config.use_sigma_reparam else ''
                self._tied_weights_keys.extend([
                    f'encoder.block.{ix}.ffn.DenseReluDense.wi_0.{indirection}weight',
                    f'encoder.block.{ix}.ffn.DenseReluDense.wi_1.{indirection}weight',
                    f'encoder.block.{ix}.ffn.DenseReluDense.wo.{indirection}weight',
                    f'encoder.block.{ix}.ffn.layer_norm.{indirection}weight',
                ])
            self.tied_ffn = T5BooruLayerFF(encoder_config)
        else:
            self.tied_ffn = None

        if config.prompt_len_avg is None:
            self.prompt_avg_key_len = None
            self.prompt_avg_key_len_population = None
            self.prompt_avg_key_len_population_max = None
            self.wants_prompt_lengths = False
        else:
            self.register_buffer(
                'prompt_avg_key_len',
                torch.tensor(
                    config.prompt_len_avg['avg_key_len_init'],
                    dtype=torch.float32,
                )
            )
            self.prompt_avg_key_len_population = config.prompt_len_avg['population_init']
            self.prompt_avg_key_len_population_max = config.prompt_len_avg['population_max']
            self.wants_prompt_lengths = True

        if config.continuation_len_avg is None:
            self.continuation_avg_key_len = None
            self.continuation_avg_key_len_population = None
            self.continuation_avg_key_len_population_max = None
            self.wants_continuation_lengths = False
        else:
            self.register_buffer(
                'continuation_avg_key_len',
                torch.tensor(
                    config.continuation_len_avg['avg_key_len_init'],
                    dtype=torch.float32,
                )
            )
            self.continuation_avg_key_len_population = config.continuation_len_avg['population_init']
            self.continuation_avg_key_len_population_max = config.continuation_len_avg['population_max']
            self.wants_continuation_lengths = True

        self.encoder = T5BooruStack(
            encoder_config,
            self.shared,
            conv_in=self.encoder_conv_in,
            tied_ffn=self.tied_ffn,
            tied_self_attn_avg_key_len=self.prompt_avg_key_len,
            # we don't pass in cross-attn key length because encoder is self-attn-only
        )

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        self.decoder = T5BooruStack(
            decoder_config,
            self.shared,
            conv_in=self.decoder_conv_in,
            # we don't pass in tied_ffn, because "One Wide FeedForward is All You Need" recommends removing decoder FFN altogether
            tied_self_attn_avg_key_len=self.continuation_avg_key_len,
            tied_cross_attn_avg_key_len=self.prompt_avg_key_len,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5BooruModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'encoder.block.0':"
            " 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5BooruModel

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5BooruModel.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5BooruModel.
        >>> # This is not needed for torch's T5BooruForMaskedLM as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if self.wants_prompt_lengths:
            if input_lengths is None:
                if attention_mask is None:
                    input_lengths = torch.full_like(input_ids, fill_value=input_ids.shape[-1], dtype=torch.int16)
                else:
                    input_lengths = attention_mask.sum(-1, keepdims=True, dtype=torch.int16)
            if self.training:
                batch_size: int = input_lengths.size(0)
                new_population: int = self.prompt_avg_key_len_population + batch_size
                self.prompt_avg_key_len.copy_((
                    self.prompt_avg_key_len.item() * self.prompt_avg_key_len_population +
                    input_lengths.sum().item()
                ) / new_population)
                # over-representing recent samples is better than overflow
                if new_population < self.prompt_avg_key_len_population_max:
                    self.prompt_avg_key_len_population = new_population

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_lengths=input_lengths,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        if self.wants_continuation_lengths:
            if decoder_input_lengths is None:
                if decoder_attention_mask is None:
                    decoder_input_lengths = torch.full_like(decoder_input_ids, fill_value=decoder_input_ids.shape[-1], dtype=torch.int16)
                else:
                    decoder_input_lengths = decoder_attention_mask.sum(-1, keepdims=True, dtype=torch.int16)
            if self.training:
                batch_size: int = decoder_input_lengths.size(0)
                new_population: int = self.continuation_avg_key_len_population + batch_size
                self.continuation_avg_key_len.copy_((
                    self.continuation_avg_key_len.item() * self.continuation_avg_key_len_population +
                    decoder_input_lengths.sum().item()
                ) / new_population)
                # over-representing recent samples is better than overflow
                if new_population < self.continuation_avg_key_len_population_max:
                    self.continuation_avg_key_len_population = new_population

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            input_lengths=decoder_input_lengths,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            encoder_input_lengths=input_lengths,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5BooruForMaskedLM(T5BooruPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    cross_entropy_loss_fn: CrossEntropyLoss
    z_loss_fn: ZLoss

    shared: Embedding
    lm_head: Linear|SReparam[Linear]
    decoder_conv_in: Optional[Conv1d|SReparam[Conv1d]]
    encoder_conv_in: Optional[Conv1d|SReparam[Conv1d]]
    tied_ffn: Optional[T5BooruLayerFF]

    prompt_avg_key_len: Optional[FloatTensor]
    prompt_avg_key_len_population: Optional[int]
    prompt_avg_key_len_population_max: Optional[int]
    continuation_avg_key_len: Optional[FloatTensor]
    continuation_avg_key_len_population: Optional[int]
    continuation_avg_key_len_population_max: Optional[int]

    # for debug (enables decoding of captions)
    vocab: Optional[Vocab] = None

    wants_prompt_lengths: bool
    wants_continuation_lengths: bool
    use_memory_efficient_attention_xformers: bool

    def __init__(self, config: T5BooruConfig):
        super().__init__(config)

        if not _xformers_available:
            logger.warn(f"xformers not found. T5Booru will fall back to torch sdp attention, which will likely fallback to the (non-accelerated) 'math' mode (due to lack of support for attention bias in its accelerated kernels).")

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        if config.encoder_conv_in:
            out_channels = config.d_model
            assert config.pad_token_id == 0, "we make use of conv1d with 'zeros' padding_mode, under the assumption that pad token will be 0. if you wish to pad with another token, then you will need to set the Conv1d's padding to none, and pad the sequence yourself with the pad token of your choosing."
            self.encoder_conv_in = Conv1d(in_channels=config.d_model, out_channels=out_channels, kernel_size=3, padding=1, bias=not config.use_sigma_reparam)
            if config.use_sigma_reparam:
                self.encoder_conv_in = SReparam(self.encoder_conv_in, **config.s_reparam_config, bias_shape=(out_channels, 1), v_shape=(1, config.d_model, 1))

            indirection = 'op.' if config.use_sigma_reparam else ''
            self._tied_weights_keys.extend([
                f'encoder.tied_conv_in.{indirection}weight',
                f'encoder.tied_conv_in.bias'
            ])
        else:
            self.encoder_conv_in = None
        
        if config.decoder_conv_in:
            out_channels = config.d_model
            assert config.pad_token_id == 0, "we make use of conv1d with 'zeros' padding_mode, under the assumption that pad token will be 0. if you wish to pad with another token, then you will need to set the Conv1d's padding to none, and pad the sequence yourself with the pad token of your choosing."
            if config.encoder_conv_in and config.tie_conv_in:
                self.decoder_conv_in = self.encoder_conv_in
            else:
                self.decoder_conv_in = Conv1d(in_channels=config.d_model, out_channels=out_channels, kernel_size=3, padding=1, bias=not config.use_sigma_reparam)
                if config.use_sigma_reparam:
                    self.decoder_conv_in = SReparam(self.decoder_conv_in, **config.s_reparam_config, bias_shape=(out_channels, 1), v_shape=(1, config.d_model, 1))

            indirection = 'op.' if config.use_sigma_reparam else ''
            self._tied_weights_keys.extend([
                f'decoder.tied_conv_in.{indirection}weight',
                f'decoder.tied_conv_in.bias'
            ])
        else:
            self.decoder_conv_in = None

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        if encoder_config.tie_encoder_ffns:
            for ix in range(encoder_config.num_layers):
                indirection = 'op.' if config.use_sigma_reparam else ''
                self._tied_weights_keys.extend([
                    f'encoder.block.{ix}.ffn.DenseReluDense.wi_0.{indirection}weight',
                    f'encoder.block.{ix}.ffn.DenseReluDense.wi_1.{indirection}weight',
                    f'encoder.block.{ix}.ffn.DenseReluDense.wo.{indirection}weight',
                    f'encoder.block.{ix}.ffn.layer_norm.{indirection}weight',
                ])
            self.tied_ffn = T5BooruLayerFF(encoder_config)
        else:
            self.tied_ffn = None

        if config.prompt_len_avg is None:
            self.prompt_avg_key_len = None
            self.prompt_avg_key_len_population = None
            self.prompt_avg_key_len_population_max = None
            self.wants_prompt_lengths = False
        else:
            self.register_buffer(
                'prompt_avg_key_len',
                torch.tensor(
                    config.prompt_len_avg['avg_key_len_init'],
                    dtype=torch.float32,
                )
            )
            self.prompt_avg_key_len_population = config.prompt_len_avg['population_init']
            self.prompt_avg_key_len_population_max = config.prompt_len_avg['population_max']
            self.wants_prompt_lengths = True

        if config.continuation_len_avg is None:
            self.continuation_avg_key_len = None
            self.continuation_avg_key_len_population = None
            self.continuation_avg_key_len_population_max = None
            self.wants_continuation_lengths = False
        else:
            self.register_buffer(
                'continuation_avg_key_len',
                torch.tensor(
                    config.continuation_len_avg['avg_key_len_init'],
                    dtype=torch.float32,
                )
            )
            self.continuation_avg_key_len_population = config.continuation_len_avg['population_init']
            self.continuation_avg_key_len_population_max = config.continuation_len_avg['population_max']
            self.wants_continuation_lengths = True

        self.encoder = T5BooruStack(
            encoder_config,
            self.shared,
            conv_in=self.encoder_conv_in,
            tied_ffn=self.tied_ffn,
            tied_self_attn_avg_key_len=self.prompt_avg_key_len,
            # we don't pass in cross-attn key length because encoder is self-attn-only
        )

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        self.decoder = T5BooruStack(
            decoder_config,
            self.shared,
            conv_in=self.decoder_conv_in,
            # we don't pass in tied_ffn, because "One Wide FeedForward is All You Need" recommends removing decoder FFN altogether
            tied_self_attn_avg_key_len=self.continuation_avg_key_len,
            tied_cross_attn_avg_key_len=self.prompt_avg_key_len,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.use_sigma_reparam:
            self.lm_head = SReparam(self.lm_head, **config.s_reparam_config)
        if config.tie_word_embeddings:
            assert not config.use_sigma_reparam, 'tie_word_embeddings is not supported when lm_head is sigma-reparameterised.'
            self._tied_weights_keys.append('lm_head.weight')
            # the assumption here is that the actual tying will be done for us by PreTrainedModel#tie_weights.
            # I haven't tested this, because T5 was like this when I got here (and doesn't typically tie embeddings).

        self.cross_entropy_loss_fn = CrossEntropyLoss(ignore_index=self.config.label_ignore_index)
        self.z_loss_fn = ZLoss(ignore_index=self.config.label_ignore_index)

        self.use_memory_efficient_attention_xformers = False

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        self.use_memory_efficient_attention_xformers = valid

        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up during
        inference. Speed up during training is not guaranteed.

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Example:

        ```py
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
        >>> model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
        """
        self.set_use_memory_efficient_attention_xformers(False)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5BooruForMaskedLM.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_lengths: Optional[torch.ShortTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_input_lengths: Optional[torch.ShortTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        z_loss: Optional[float] = 1e-4
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5BooruForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5BooruForMaskedLM.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # [[self.vocab.tokens[token_ix] for token_ix in caption] for caption in input_ids]

        if self.wants_prompt_lengths:
            if input_lengths is None:
                if attention_mask is None:
                    input_lengths = torch.full_like(input_ids, fill_value=input_ids.shape[-1], dtype=torch.int16)
                else:
                    input_lengths = attention_mask.sum(-1, keepdims=True, dtype=torch.int16)
            if self.training:
                batch_size: int = input_lengths.size(0)
                new_population: int = self.prompt_avg_key_len_population + batch_size
                self.prompt_avg_key_len.copy_((
                    self.prompt_avg_key_len.item() * self.prompt_avg_key_len_population +
                    input_lengths.sum().item()
                ) / new_population)
                # over-representing recent samples is better than overflow
                if new_population < self.prompt_avg_key_len_population_max:
                    self.prompt_avg_key_len_population = new_population

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_lengths=input_lengths,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            assert decoder_attention_mask is None, "we could create decoder_input_ids by shifting your labels, but you have provided a decoder_attention_mask, and we don't know whether it's a mask for the labels (which we don't need) or a mask for the decoder_input_ids (which you've never seen). you should pass in decoder_input_ids and its corresponding decoder_attention_mask."
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids: LongTensor = self._shift_right(labels.int())

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        if self.wants_continuation_lengths:
            if decoder_input_lengths is None:
                if decoder_attention_mask is None:
                    decoder_input_lengths = torch.full_like(decoder_input_ids, fill_value=decoder_input_ids.shape[-1], dtype=torch.int16)
                else:
                    decoder_input_lengths = decoder_attention_mask.sum(-1, keepdims=True, dtype=torch.int16)
            if self.training:
                batch_size: int = decoder_input_lengths.size(0)
                new_population: int = self.continuation_avg_key_len_population + batch_size
                self.continuation_avg_key_len.copy_((
                    self.continuation_avg_key_len.item() * self.continuation_avg_key_len_population +
                    decoder_input_lengths.sum().item()
                ) / new_population)
                # over-representing recent samples is better than overflow
                if new_population < self.continuation_avg_key_len_population_max:
                    self.continuation_avg_key_len_population = new_population

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            input_lengths=decoder_input_lengths,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            encoder_input_lengths=input_lengths,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits: FloatTensor = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable PP
            labels: LongTensor = labels.to(lm_logits.device, dtype=torch.long)

            # z_loss encourages the logits:
            # - to not drift too far from zero (which can cause unacceptable roundoff errors in bfloat16)
            # - to be normalized log-probabilities
            if z_loss is None or z_loss == 0.:
                loss = self.cross_entropy_loss_fn(lm_logits.flatten(end_dim=-2), labels.flatten())
            else:
                loss = self.z_loss_fn(lm_logits, labels, z_loss=z_loss)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # commented-out because although this would be more efficient than padding just-in-time,
        # it gets a lot more complicated once we start concatenating in an unmasked past_key_values.
        # if self.use_memory_efficient_attention_xformers:
        #     pad_to_multiple = 8
        #     if input_ids.shape[-1] % pad_to_multiple != 0:
        #         if decoder_attention_mask is None:
        #             decoder_attention_mask = (torch.arange(ceil_to_multiple(input_ids.shape[-1], pad_to_multiple), device=input_ids.device) < input_ids.shape[-1]).expand(input_ids.shape[0], -1)
        #         else:
        #             decoder_attention_mask = pad(decoder_attention_mask, pad=(0, remaining_to_multiple(decoder_attention_mask.shape[-1], pad_to_multiple)), value=False)
        #         input_ids = pad(input_ids, pad=(0, remaining_to_multiple(input_ids.shape[-1], pad_to_multiple)), value=self.config.pad_token_id)

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5BooruEncoderModel(T5BooruPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    lm_head: Linear|SReparam[Linear]
    shared: Embedding
    decoder_conv_in: Optional[Conv1d|SReparam[Conv1d]]
    encoder_conv_in: Optional[Conv1d|SReparam[Conv1d]]
    tied_ffn: Optional[T5BooruLayerFF]
    prompt_avg_key_len: Optional[FloatTensor]
    prompt_avg_key_len_population: Optional[int]
    prompt_avg_key_len_population_max: Optional[int]

    wants_prompt_lengths: bool

    def __init__(self, config: T5BooruConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        if config.encoder_conv_in:
            out_channels = config.d_model
            assert config.pad_token_id == 0, "we make use of conv1d with 'zeros' padding_mode, under the assumption that pad token will be 0. if you wish to pad with another token, then you will need to set the Conv1d's padding to none, and pad the sequence yourself with the pad token of your choosing."
            self.encoder_conv_in = Conv1d(in_channels=config.d_model, out_channels=out_channels, kernel_size=3, padding=1, bias=not config.use_sigma_reparam)
            if config.use_sigma_reparam:
                self.encoder_conv_in = SReparam(self.encoder_conv_in, **config.s_reparam_config, bias_shape=(out_channels, 1), v_shape=(1, config.d_model, 1))

            indirection = 'op.' if config.use_sigma_reparam else ''
            self._tied_weights_keys.extend([
                f'encoder.tied_conv_in.{indirection}weight',
                f'encoder.tied_conv_in.bias'
            ])
        else:
            self.encoder_conv_in = None

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        if encoder_config.tie_encoder_ffns:
            for ix in range(encoder_config.num_layers):
                indirection = 'op.' if config.use_sigma_reparam else ''
                self._tied_weights_keys.extend([
                    f'encoder.block.{ix}.ffn.DenseReluDense.wi_0.{indirection}weight',
                    f'encoder.block.{ix}.ffn.DenseReluDense.wi_1.{indirection}weight',
                    f'encoder.block.{ix}.ffn.DenseReluDense.wo.{indirection}weight',
                    f'encoder.block.{ix}.ffn.layer_norm.{indirection}weight',
                ])
            self.tied_ffn = T5BooruLayerFF(encoder_config)
        else:
            self.tied_ffn = None

        if config.prompt_len_avg is None:
            self.prompt_avg_key_len = None
            self.prompt_avg_key_len_population = None
            self.prompt_avg_key_len_population_max = None
            self.wants_prompt_lengths = False
        else:
            self.register_buffer(
                'prompt_avg_key_len',
                torch.tensor(
                    config.prompt_len_avg['avg_key_len_init'],
                    dtype=torch.float32,
                )
            )
            self.prompt_avg_key_len_population = config.prompt_len_avg['population_init']
            self.prompt_avg_key_len_population_max = config.prompt_len_avg['population_max']
            self.wants_prompt_lengths = True

        self.encoder = T5BooruStack(
            encoder_config,
            self.shared,
            conv_in=self.encoder_conv_in,
            tied_ffn=self.tied_ffn,
            tied_self_attn_avg_key_len=self.prompt_avg_key_len,
        )

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5BooruEncoderModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5BooruEncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5BooruEncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.wants_prompt_lengths:
            if input_lengths is None:
                if attention_mask is None:
                    input_lengths = torch.full_like(input_ids, fill_value=input_ids.shape[-1], dtype=torch.int16)
                else:
                    input_lengths = attention_mask.sum(-1, keepdims=True, dtype=torch.int16)
            if self.training:
                batch_size: int = input_lengths.size(0)
                new_population: int = self.prompt_avg_key_len_population + batch_size
                self.prompt_avg_key_len.copy_((
                    self.prompt_avg_key_len.item() * self.prompt_avg_key_len_population +
                    input_lengths.sum().item()
                ) / new_population)
                # over-representing recent samples is better than overflow
                if new_population < self.prompt_avg_key_len_population_max:
                    self.prompt_avg_key_len_population = new_population

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

