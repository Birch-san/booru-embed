from transformers import TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from logging import getLogger
from torch import nn
from torch.optim import Optimizer
from typing import Set
logger = getLogger(__name__)

from src.model.modeling_t5_booru import T5BooruForMaskedLM

def create_optimizer(
    model: T5BooruForMaskedLM,
    train_args: TrainingArguments,
) -> Optimizer:
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.

    Fork of HF transformers' create_optimizer:
    https://github.com/huggingface/transformers/blob/0a55d9f7376f72ad3ff296d4249840021b03bcc4/src/transformers/trainer.py#L954
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameter_set: Set[str] = {
        name for name in decay_parameters if (
            'bias' not in name and
            # SReparam weight should not be decayed. instead, SReparam.gamma should be decayed
            not name.endswith('.op.weight')
        )
    }
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameter_set and p.requires_grad)
            ],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameter_set and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(train_args)

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                logger.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")

    return optimizer