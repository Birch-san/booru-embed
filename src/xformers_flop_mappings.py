from torch import Size
from typing import NamedTuple, Literal, Optional
from xformers.ops import memory_efficient_attention, MemoryEfficientAttentionCutlassOp

cutlass_fwd, cutlass_bwd = MemoryEfficientAttentionCutlassOp

class TensorMeta:
  shape: Size
  ndim: int
  def __init__(self, shape: Size) -> None:
    self.shape = shape
    self.ndim = len(shape)

class CutlassFwdOutShape(NamedTuple):
  # b t h c
  attn_scores: Size
  # b h t
  logsumexp: Size
  rng_seed: int
  rng_offset: int

def cutlass_fwd_flop(
  query: Size,
  key: Size,
  value: Size,
  attn_bias: Optional[Size],
  seqstart_q: Optional[Size],
  seqstart_k: Optional[Size],
  max_seqlen_q: int,
  dropout_p: float,
  compute_logsumexp: bool,
  custom_mask_type: Literal[0, 1, 2],
  scale: Optional[float],
  seqlen_k: Optional[bool],
  out_shape: CutlassFwdOutShape,
):
  assert seqstart_q is None and seqstart_k is None, "Cannot compute flops due to use of BlockDiagonalMask/BlockDiagonalCausalWithOffsetPaddedKeysMask. we need the tensor information contained in seqstart_q and seqstart_k, but FlopCounter's torch_dispatch only gave us the shapes, not the data."
  # this thing expects to receive tensors, but we don't have any
  # fortunately it's only interested in their shapes and dims
  return cutlass_fwd.operator_flop(
    TensorMeta(query),
    TensorMeta(key),
    TensorMeta(value),
    attn_bias, # unused
    seqstart_q,
    seqstart_k,
    max_seqlen_q, # unused
    compute_logsumexp, # unused
    custom_mask_type,
  )

class CutlassBwdOutShape(NamedTuple):
  # b t h c
  grad_query: Size
  grad_key: Size
  grad_value: Size
  # shouldn't require grad, so None (except perhaps if you made a mistake)
  grad_bias: Optional[Size]

def cutlass_bwd_flop(
  grad: Size,
  query: Size,
  key: Size,
  value: Size,
  attn_bias: Optional[Size],
  cu_seqlens_q: Optional[Size],
  cu_seqlens_k: Optional[Size],
  max_seqlen_q: int,
  max_seqlen_k: int,
  logsumexp: Size,
  output: Size,
  dropout_p: float,
  rng_seed: int,
  rng_offset: int,
  custom_mask_type: Literal[0, 1, 2],
  scale: Optional[float],
  num_splits_key: int,
  out_shape: CutlassBwdOutShape,
):
  assert cu_seqlens_q is None and cu_seqlens_k is None, "Cannot compute flops due to use of BlockDiagonalMask/BlockDiagonalCausalWithOffsetPaddedKeysMask. we need the tensor information contained in seqstart_q and seqstart_k, but FlopCounter's torch_dispatch only gave us the shapes, not the data."
  # this thing expects to receive tensors, but we don't have any
  # fortunately it's only interested in their shapes and dims
  return cutlass_bwd.operator_flop(
    grad, # unused
    TensorMeta(query),
    TensorMeta(key),
    TensorMeta(value),
    attn_bias, # unused
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q, # unused
    max_seqlen_k, # unused
    logsumexp, # unused
    output, # unused
    dropout_p, # unused
    rng_seed, # unused
    rng_offset, # unused
    custom_mask_type,
    scale, # unused
  )