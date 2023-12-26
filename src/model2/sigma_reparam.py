from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d, Conv2d

from torch import Tensor, FloatTensor
from typing import Optional, TypeAlias, Sequence, Generic, TypeVar

# more than these are supported, but these are the ones I care enough to mention in the types
SReparamSupportedModule: TypeAlias = Linear | Conv1d | Conv2d
M = TypeVar("M", bound=SReparamSupportedModule)

# TODO: consider replacing with Apple's https://github.com/apple/ml-sigma-reparam
class SReparam(nn.Module, Generic[M]):
    """
    σReparam implementation by Katherine Crowson
    Stabilizing Transformer Training by Preventing Attention Entropy Collapse
    https://arxiv.org/abs/2303.06296

    if you are using gradient checkpointing: be sure to enable use_rentrant=True

    some changes by Alex Birch:
    - type hints
    - addcmul fusion
    - register_v_during_construction + init_all_via_trace, to try and improve support for loading saved checkpoints via HF transformers
    """
    op: M
    n_iters: int
    n_iters_init: int
    eps: float
    register_v_during_construction: bool
    # gamma, but we can't call it gamma because (sigh):
    # https://github.com/huggingface/transformers/blob/8e3980a290acc6d2f8ea76dba111b9ef0ef00309/src/transformers/modeling_utils.py#L3355
    g: nn.Parameter
    bias: Optional[nn.Parameter]
    heads: Optional[int]

    sigma: FloatTensor
    # type hint for (potentially) lateinit buffer
    v: FloatTensor

    def __init__(
        self,
        op: M,
        v_shape: Optional[Sequence[int]] = None,
        bias_shape: Optional[Sequence[int]] = None,
        n_iters=1,
        n_iters_init=15,
        eps=1e-12,
        learn_gamma=True,
        # validates that v buffer is registered on construction instead of relying on late-initialization
        # via init_all_via_trace(). this ensures that any state_dict we save, can be loaded
        # by HF transformers load_pretrained() (which loads state_dict entries immediately after
        # construction, without leaving any opportunity to register buffers via delayed-init).
        register_v_during_construction=True,
        heads: Optional[int] = None,
    ):
        super().__init__()
        self.op = op
        self.n_iters = n_iters
        self.n_iters_init = max(n_iters, n_iters_init)
        self.eps = eps
        self.g = nn.Parameter(torch.ones(() if heads is None else (heads,)), requires_grad=learn_gamma)
        self.bias = nn.Parameter(torch.zeros(bias_shape)) if bias_shape is not None else None
        self.heads = heads
        if heads is not None:
            assert register_v_during_construction, "you have requested multi-headed SReparam and late-initialized v buffer, but we have not implemented inferring of shape of multi-headed v via tracing. you will have to register_v_during_construction=True for now."
            assert isinstance(op, Linear), "multi-headed SReparam is only implemented for Linear layers."
            # assert self.bias is None, "multi-headed SReparam is not implemented for layers with bias"
        self.register_v_during_construction = register_v_during_construction
        if register_v_during_construction:
            if v_shape is None:
                if isinstance(op, Linear):
                    v_shape = (op.in_features,) if heads is None else (heads, op.in_features)
                else:
                    raise ValueError('you have specified `register_v_during_construction`, but we were unable to infer v_shape from your op (we can only do this for Linear).')
            self.register_buffer('v', torch.randn(v_shape))
        self.register_buffer('sigma', torch.ones(() if heads is None else (heads,)))

    # TODO: power iteration should be done in fp32 (even tf32 isn't enough)
    @torch.no_grad()
    def update_(self, n_iters: Optional[int] = None) -> None:
        n_iters: int = n_iters or self.n_iters
        v: Tensor = self.v
        if self.heads is not None:
            pass
        dim: Optional[int] = None if self.heads is None else -1
        keepdim: bool = dim is not None
        for _ in range(n_iters):
            u: Tensor = self.op(v)
            u = u / u.norm(dim=dim, keepdim=keepdim).clamp_min(self.eps)
            _, v = torch.autograd.functional.vjp(self.op, v, u)
            v = v / v.norm(dim=dim, keepdim=keepdim).clamp_min(self.eps)
        self.sigma.copy_(torch.sum(u * self.op(v), dim=dim))
        self.v.copy_(v)

    @classmethod
    def update_all_(cls, module: nn.Module, n_iters: Optional[int] = None) -> None:
        for child in module.children():
            if isinstance(child, cls):
                # TODO: avoid doubly-updating tied layers!
                child.update_(n_iters)
            else:
                cls.update_all_(child, n_iters)

    @torch.no_grad()
    def init_(self, shape: Sequence[int], dtype: torch.dtype, device: torch.device|str) -> None:
        if hasattr(self, 'v'):
            assert self.register_v_during_construction
            assert self.v.device == device
            assert self.sigma.device == device
            if self.v.dtype != dtype:
                self.v.data = self.v.data.to(dtype)
            if self.sigma.dtype != dtype:
                self.sigma.data = self.sigma.data.to(dtype)
        else:
            assert not self.register_v_during_construction
            self.register_buffer('v', torch.randn(shape, dtype=dtype, device=device))
        self.update_(self.n_iters_init)

    @classmethod
    def init_all_via_trace(cls, module: nn.Module, *args, **kwargs) -> None:
        module(*args, **kwargs)
    
    @classmethod
    def init_all_statically(cls, module: nn.Module) -> None:
        for mod in module.modules():
            if isinstance(mod, SReparam):
                dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else mod.v.dtype
                device = mod.v.device
                mod.init_(mod.n_iters_init, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # note: this condition may cost you a graph break if you are using torch.compile().
        # if you are using late-init: prefer to run your init_all_via_trace() *before* torch.compile(),
        # so that the noattr branch never needs to be compiled.
        if not hasattr(self, 'v') and self.training:
            self.init_(x.shape, x.dtype, x.device)
        y: Tensor = self.op(x)
        if self.heads is None:
            g = self.g
            sigma = self.sigma
        else:
            y = y.unflatten(dim=-1, sizes=(self.heads, -1))
            broadcast_shape = [1] * y.ndim
            broadcast_shape[-2] = self.heads
            g = self.g.view(broadcast_shape)
            sigma = self.sigma.view(broadcast_shape)
        # it's possible to use torch.addcmul as a fastpath to fuse the hadamard and bias
        # in fact if we peek at what the underlying op is, we could probably fuse all the way
        # into that. but there are a lot of combinations to support; hopefully torch.compile does this all for us.
        y *= (g / sigma).to(y.dtype)
        if self.heads is not None:
            y = y.flatten(start_dim=-2)
        if self.bias is not None:
            y += self.bias.to(y.dtype)
        return y
