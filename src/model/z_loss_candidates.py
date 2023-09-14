from torch import FloatTensor, LongTensor
from torch.nn import LogSoftmax, NLLLoss, CrossEntropyLoss
from torch.nn.functional import one_hot

# labels2 = labels.where(labels != -100, 32175)
# labels2_flat = labels2.flatten()

# self.cross_entropy_loss_fn(lm_logits.flatten(end_dim=-2), labels2_flat)
# old_z_loss_fn(lm_logits, labels2, labels2_flat)
# older_z_loss_fn(lm_logits, labels2)
# new_z_loss_fn(lm_logits, labels2, labels2_flat)
# ce_fn(lm_logits, labels2_flat)


# when there's no -100 tokens, cross-entropy is 101.8485:
# - all 3 z_loss fns are equivalent
# - all 3 z_loss fns are equivalent to ce_fn when z_loss=0

# at float() precision,
#  self.cross_entropy_loss_fn(lm_logits.flatten(end_dim=-2).float(), labels2_flat)
# is identical to:
#   ce_fn(lm_logits, labels2_flat)

## now, with -100
# self.cross_entropy_loss_fn(lm_logits.flatten(end_dim=-2).float(), labels.flatten())
#   93.0354
# new_z_loss_fn(lm_logits, labels, labels_flat, z_loss=0)
#   49.1343
# old_z_loss_fn(lm_logits, labels, labels_flat, z_loss=0)
#   49.1343

# let's just crop out the -100 instead
# self.cross_entropy_loss_fn(lm_logits[0,:-19].float(), labels[0,:-19])
#   93.6663
# self.cross_entropy_loss_fn(lm_logits[0,:].float(), labels[0,:])
#   93.6663
# okay, we trust CE
# ce_fn(lm_logits[0,:], labels[0,:])
#   93.6663
# ce_fn(lm_logits[0,:-19], labels[0,:-19])
#   93.6663
# okay, we trust both CEs
# old_z_loss_fn(lm_logits[0,:], labels[0,:], labels[0,:].flatten(), z_loss=0)
#   49.1748
# old_z_loss_fn(lm_logits[0,:-19], labels[0,:-19], labels[0,:-19].flatten(), z_loss=0)
#   93.6663
# new_z_loss_fn(lm_logits[0,:], labels[0,:], labels[0,:].flatten(), z_loss=0)
#   49.1748
# new_z_loss_fn(lm_logits[0,:-19], labels[0,:-19], labels[0,:-19].flatten(), z_loss=0)
#   93.6663
# alright, so the problem is the masking on my z_loss functions, but my CE fn is fine.


# I think this is bad
# old_z_loss_fn(lm_logits, labels, labels_flat)
#   tensor(50.1017, device='cuda:0', grad_fn=<MeanBackward0>)
def old_z_loss_fn(lm_logits: FloatTensor, labels: LongTensor, labels_flat: FloatTensor, z_loss = .0001) -> FloatTensor:
  nll_loss_no_reduce_fn = NLLLoss(reduction='none')

  log_z: FloatTensor = lm_logits.logsumexp(dim=-1)
  log_softmax: FloatTensor = lm_logits - log_z.unsqueeze(-1)

  nll_inputs: FloatTensor = log_softmax.flatten(end_dim=-2)

  loss: FloatTensor = nll_loss_no_reduce_fn(nll_inputs, labels_flat)
  loss = loss.unflatten(0, labels.shape)
  loss += z_loss * log_z ** 2
  nonignored_token_count: LongTensor = labels.numel() - (labels == -100).sum()
  loss = loss.sum() / nonignored_token_count
  return loss

# I'm getting CUDA errors when I run one_hot, but this used to work, right?
def older_z_loss_fn(lm_logits: FloatTensor, labels: LongTensor, z_loss = .0001) -> FloatTensor:
  log_z: FloatTensor = lm_logits.logsumexp(dim=-1)
  log_softmax: FloatTensor = lm_logits - log_z.unsqueeze(-1)

  labels_oh: LongTensor = one_hot(labels.long(), lm_logits.size(-1))
  loss: FloatTensor = log_softmax.mul(labels_oh).sum(dim=-1).neg()
  loss += z_loss * log_z ** 2
  loss = loss.mean()
  return loss

def new_z_loss_fn(lm_logits: FloatTensor, labels: LongTensor, labels_flat: FloatTensor, z_loss = .0001) -> FloatTensor:
  nll_loss_fn = NLLLoss(reduction='none')

  log_z: FloatTensor = lm_logits.logsumexp(dim=-1)
  log_softmax: FloatTensor = lm_logits - log_z.unsqueeze(-1)
  loss: FloatTensor = nll_loss_fn(log_softmax.flatten(end_dim=-2), labels_flat)
  loss = loss.unflatten(0, labels.shape)
  masked_log_z: FloatTensor = log_z.where(labels != -100, 0)
  loss += z_loss * masked_log_z ** 2
  nonignored_token_count: LongTensor = labels.numel() - (labels == -100).sum()
  loss = loss.sum() / nonignored_token_count
  return loss

# sems equivalent to CrossEntropy
# ce_fn(lm_logits, labels_flat)
#   tensor(93.0354, device='cuda:0', grad_fn=<NllLossBackward0>)
# self.cross_entropy_loss_fn(lm_logits.flatten(end_dim=-2), labels_flat)
#   tensor(93.0533, device='cuda:0', grad_fn=<NllLossBackward0>)
def ce_fn(lm_logits: FloatTensor, labels_flat: FloatTensor) -> FloatTensor:
  log_softmax_fn = LogSoftmax(dim=-1)
  nll_loss_reduce_fn = NLLLoss()

  log_softmax: FloatTensor = log_softmax_fn(lm_logits)

  nll_inputs: FloatTensor = log_softmax.flatten(end_dim=-2)

  loss: FloatTensor = nll_loss_reduce_fn(nll_inputs, labels_flat)
  return loss

def ce_fn_half(lm_logits: FloatTensor, labels_flat: FloatTensor) -> FloatTensor:
  log_softmax_fn = LogSoftmax(dim=-1)
  nll_loss_reduce_fn = NLLLoss()
  
  orig_dtype = lm_logits.dtype
  log_softmax: FloatTensor = log_softmax_fn(lm_logits)
  log_softmax: FloatTensor = log_softmax.to(orig_dtype)

  nll_inputs: FloatTensor = log_softmax.flatten(end_dim=-2)

  loss: FloatTensor = nll_loss_reduce_fn(nll_inputs, labels_flat)
  return loss