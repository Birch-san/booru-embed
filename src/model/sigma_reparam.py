import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d, Conv2d
from torch import Tensor
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

    + type hints and addcmul fusion added by Alex Birch
    """
    op: M
    n_iters: int
    n_iters_init: int
    eps: float
    # gamma, but we can't call it gamma because (sigh):
    # https://github.com/huggingface/transformers/blob/8e3980a290acc6d2f8ea76dba111b9ef0ef00309/src/transformers/modeling_utils.py#L3355
    g: nn.Parameter
    bias: Optional[nn.Parameter]

    # type hints for buffers registered during init_()
    sigma: Tensor
    v: Tensor

    def __init__(
        self,
        op: M,
        bias_shape: Optional[Sequence[int]] = None,
        n_iters=1,
        n_iters_init=15,
        eps=1e-12,
        learn_gamma=True,
    ):
        super().__init__()
        self.op = op
        self.n_iters = n_iters
        self.n_iters_init = max(n_iters, n_iters_init)
        self.eps = eps
        self.g = nn.Parameter(torch.ones(()), requires_grad=learn_gamma)
        self.bias = nn.Parameter(torch.zeros(bias_shape)) if bias_shape is not None else None

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
        self.register_buffer("v", torch.randn(shape, dtype=dtype, device=device))
        self.register_buffer("sigma", torch.ones((), dtype=self.g.dtype, device=device))
        self.update_(self.n_iters_init)

    @classmethod
    def init_all_(cls, module: nn.Module, *args, **kwargs) -> None:
        module(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if not hasattr(self, "sigma"):
            self.init_(x.shape, x.dtype, x.device)
        y: Tensor = self.op(x)
        if self.bias is None:
            return y * (self.g / self.sigma).to(y.dtype)
        return torch.addcmul(self.bias.to(y.dtype), y, self.g.to(y.dtype), value=(1 / self.sigma).to(y.dtype))
