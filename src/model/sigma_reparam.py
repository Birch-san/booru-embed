import torch
import torch.nn as nn

class SReparam(nn.Module):
    """
    σReparam implementation by Katherine Crowson
    Stabilizing Transformer Training by Preventing Attention Entropy Collapse
    https://arxiv.org/abs/2303.06296
    """
    def __init__(self, op, bias_shape=None, n_iters=1, n_iters_init=15, eps=1e-12, learn_gamma=True):
        super().__init__()
        self.op = op
        self.n_iters = n_iters
        self.n_iters_init = max(n_iters, n_iters_init)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(()), requires_grad=learn_gamma)
        self.bias = nn.Parameter(torch.zeros(bias_shape)) if bias_shape is not None else None

    @torch.no_grad()
    def update_(self, n_iters=None):
        n_iters = n_iters or self.n_iters
        v = self.v
        for _ in range(n_iters):
            u = self.op(v)
            u = u / u.norm().clamp_min(self.eps)
            _, v = torch.autograd.functional.vjp(self.op, v, u)
            v = v / v.norm().clamp_min(self.eps)
        self.sigma.copy_(torch.sum(u * self.op(v)))
        self.v.copy_(v)

    @classmethod
    def update_all_(cls, module, n_iters=None):
        for child in module.children():
            if isinstance(child, cls):
                child.update_(n_iters)
            else:
                cls.update_all_(child, n_iters)

    @torch.no_grad()
    def init_(self, shape, dtype, device):
        self.register_buffer("v", torch.randn(shape, dtype=dtype, device=device))
        self.register_buffer("sigma", torch.ones((), dtype=self.gamma.dtype, device=device))
        self.update_(self.n_iters_init)

    @classmethod
    def init_all_(cls, module, *args, **kwargs):
        module(*args, **kwargs)

    def forward(self, x):
        if not hasattr(self, "sigma"):
            self.init_(x.shape, x.dtype, x.device)
        y = self.op(x)
        y = y * (self.gamma / self.sigma.clone()).to(y.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y