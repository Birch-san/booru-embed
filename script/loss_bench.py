import torch
from torch import FloatTensor, LongTensor
from torch.nn import NLLLoss, Module
from typing import Protocol, NamedTuple, Callable
import time

class IncrementLoss(Protocol):
    def __call__(self, loss: FloatTensor, masked_log_z: FloatTensor, z_loss: float) -> None: ...

class Inputs(NamedTuple):
    lm_logits: FloatTensor
    labels: LongTensor

class ComputeZLoss(Module):
    increment_loss: IncrementLoss
    z_loss: float
    label_ignore_index: int
    nll_loss_fn: NLLLoss

    def __init__(
        self,
        increment_loss: IncrementLoss,
        z_loss: float = 0.0001,
        label_ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.increment_loss = increment_loss
        self.nll_loss_fn = NLLLoss(reduction='none', ignore_index=label_ignore_index)
        self.z_loss = z_loss
        self.label_ignore_index = label_ignore_index

    def __call__(self, lm_logits: FloatTensor, labels: LongTensor) -> FloatTensor:
        log_z: FloatTensor = lm_logits.logsumexp(dim=-1)
        log_softmax: FloatTensor = lm_logits - log_z.unsqueeze(-1)
        loss: FloatTensor = self.nll_loss_fn(log_softmax.flatten(end_dim=-2), labels.flatten())
        loss = loss.unflatten(0, labels.shape)
        masked_log_z: FloatTensor = log_z.where(labels != self.label_ignore_index, 0)
        self.increment_loss(loss, masked_log_z, self.z_loss)
        nonignored_token_count: LongTensor = labels.numel() - (labels == self.label_ignore_index).sum()
        loss = loss.sum() / nonignored_token_count
        loss.backward()
        return loss

def arithmetic_inc(loss: FloatTensor, masked_log_z: FloatTensor, z_loss: float) -> None:
    loss += z_loss * masked_log_z ** 2

def addcmul_inc(loss: FloatTensor, masked_log_z: FloatTensor, z_loss: float) -> FloatTensor:
    loss.addcmul_(masked_log_z, masked_log_z, value=z_loss)

device = torch.device('cuda')
batch_size = 256
seq_len = 128
vocab_size = 32000
make_inputs: Callable[[], Inputs] = lambda: Inputs(
    lm_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.bfloat16, requires_grad=True),
    labels = (torch.rand(batch_size, seq_len, device=device) * vocab_size).long(),
)

def bench(f: Callable[[], None], name=None, iters=100, warmup=5, display=True, profile=False) -> str:
    for _ in range(warmup):
        f()
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()
    us_per_iter = (time.perf_counter()-begin)*1e6/iters
    if name is None:
        res = us_per_iter
    else:
        res= f"{name}: {us_per_iter:.2f}us"
    if display:
        print(res)
    return res

arith_zloss = ComputeZLoss(arithmetic_inc).to(device)
addcmul_zloss = ComputeZLoss(addcmul_inc).to(device)

arith_zloss_compile_default = torch.compile(ComputeZLoss(arithmetic_inc).to(device))
addcmul_zloss_default = torch.compile(ComputeZLoss(addcmul_inc).to(device))

arith_zloss_compile_reduce_overhead = torch.compile(ComputeZLoss(arithmetic_inc).to(device), mode='reduce-overhead')
addcmul_zloss_reduce_overhead = torch.compile(ComputeZLoss(addcmul_inc).to(device), mode='reduce-overhead')

arith_zloss_compile_max_autotune = torch.compile(ComputeZLoss(arithmetic_inc).to(device), mode='max-autotune')
addcmul_zloss_max_autotune = torch.compile(ComputeZLoss(addcmul_inc).to(device), mode='max-autotune')

# arithmetic  49933.31us
# add_cmul    49932.88us
bench(lambda: arith_zloss(*make_inputs()), "arithmetic z_loss")
bench(lambda: addcmul_zloss(*make_inputs()), "addcmul z_loss")

# arithmetic  15697.97us
# add_cmul    15648.07us
bench(lambda: arith_zloss_compile_default(*make_inputs()), "arithmetic z_loss torch.compile (default)")
bench(lambda: addcmul_zloss_default(*make_inputs()), "addcmul z_loss torch.compile (default)")

# arithmetic  20247.69us
# add_cmul    20259.84us
bench(lambda: arith_zloss_compile_reduce_overhead(*make_inputs()), "arithmetic z_loss torch.compile (reduce-overhead)")
bench(lambda: addcmul_zloss_reduce_overhead(*make_inputs()), "addcmul z_loss torch.compile (reduce-overhead)")

# arithmetic  20257.20us
# add_cmul    20259.15us
bench(lambda: arith_zloss_compile_max_autotune(*make_inputs()), "arithmetic z_loss torch.compile (max-autotune)")
bench(lambda: addcmul_zloss_max_autotune(*make_inputs()), "addcmul z_loss torch.compile (max-autotune)")