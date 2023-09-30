from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d, Conv2d

from torch import Tensor, FloatTensor
from typing import Optional, TypeAlias, Sequence, Generic, TypeVar

# more than these are supported, but these are the ones I care enough to mention in the types
SReparamSupportedModule: TypeAlias = Linear | Conv1d | Conv2d
M = TypeVar("M", bound=SReparamSupportedModule)

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
    ):
        super().__init__()
        self.op = op
        self.n_iters = n_iters
        self.n_iters_init = max(n_iters, n_iters_init)
        self.eps = eps
        self.g = nn.Parameter(torch.ones(()), requires_grad=learn_gamma)
        self.bias = nn.Parameter(torch.zeros(bias_shape)) if bias_shape is not None else None
        self.register_v_during_construction = register_v_during_construction
        if register_v_during_construction:
            if v_shape is None:
                if isinstance(op, Linear):
                    v_shape = (op.in_features,)
                else:
                    raise ValueError('you have specified `register_v_during_construction`, but we were unable to infer v_shape from your op (we can only do this for Linear).')
            self.register_buffer('v', torch.randn(v_shape))
        self.register_buffer('sigma', torch.ones(()))

    @torch.no_grad()
    def update_(self, n_iters: Optional[int] = None) -> None:
        n_iters: int = n_iters or self.n_iters
        v: Tensor = self.v
        for _ in range(n_iters):
            u: Tensor = self.op(v)
            u = u / u.norm().clamp_min(self.eps)
            _, v = torch.autograd.functional.vjp(self.op, v, u)
            v = v / v.norm().clamp_min(self.eps)
        self.sigma.copy_(torch.sum(u * self.op(v)))
        self.v.copy_(v)

    @classmethod
    def update_all_(cls, module: nn.Module, n_iters: Optional[int] = None) -> None:
        for child in module.children():
            if isinstance(child, cls):
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
        # TODO: fastpath to fuse hadamards into operation
        if self.bias is None:
            return y * (self.g / self.sigma).to(y.dtype)
        return torch.addcmul(self.bias.to(y.dtype), y, self.g.to(y.dtype), value=(1 / self.sigma).to(y.dtype))
