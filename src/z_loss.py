from torch import FloatTensor, LongTensor
from torch.nn import Module, NLLLoss
from typing import Literal, TypeAlias, Optional

Reduction: TypeAlias = Literal['mean', 'sum', 'none']

class ZLoss(Module):
    """
    ZLoss is a modified CrossEntropyLoss.
    when z_loss=0: they are equivalent.

    z_loss encourages the logits:
    - to not drift too far from zero (which can cause unacceptable roundoff errors in bfloat16)
    - to be normalized log-probabilities

    based on t5x and mesh_tensorflow implementations:
    https://github.com/google-research/t5x/blob/77d2624e65799e3bea15586eb1d3fe7c63477a92/t5x/models.py#L738
    https://github.com/google-research/t5x/blob/0728d8429041d6c6e75077334e76eb2370c6057b/t5x/losses.py#L50
    https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
    """
    ignore_index: int
    nll_loss_fn: NLLLoss
    reduction: Reduction
    z_loss: float
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: Reduction = 'mean',
        z_loss: float = 1e-4,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss_fn = NLLLoss(reduction='none', ignore_index=ignore_index)
        self.z_loss = z_loss
    
    def forward(self, logits: FloatTensor, labels: LongTensor, z_loss: Optional[float] = 1e-4) -> FloatTensor:
        logits_dtype = logits.dtype
        log_z: FloatTensor = logits.logsumexp(dim=-1)
        log_softmax: FloatTensor = logits - log_z.unsqueeze(-1)
        # logsumexp increases our precision to float32.
        # we return to original precision after computing log_softmax, to take the approach I believe CrossEntropyLoss is using —
        # this seems to ensure that we get the same result (when z_loss=0).
        log_softmax = log_softmax.to(logits_dtype)

        loss: FloatTensor = self.nll_loss_fn(log_softmax.flatten(end_dim=-2), labels.flatten())
        loss = loss.unflatten(0, labels.shape)
        masked_log_z: FloatTensor = log_z.where(labels != self.ignore_index, 0)
        # this addcmul_ is just a (hopefully faster) way to express:
        #   loss += z_loss * masked_log_z ** 2
        loss.addcmul_(masked_log_z, masked_log_z, value=self.z_loss if z_loss is None else z_loss)
        if self.reduction == 'none':
            return loss
        loss = loss.sum()
        if self.reduction == 'sum':
            return loss
        nonignored_token_count: LongTensor = labels.numel() - (labels == self.ignore_index).sum()
        loss /= nonignored_token_count
        return loss