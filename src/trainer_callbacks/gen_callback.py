from transformers import (
	GenerationConfig,
	TrainerCallback,
	TrainingArguments,
	TrainerControl,
	TrainerState,
	T5ForConditionalGeneration,
)
from dataclasses import dataclass, field
import types
from typing import Iterable, Iterator, Dict, Literal
from torch import LongTensor, inference_mode
from itertools import tee, cycle
from enum import Enum, auto
import torch
from contextlib import nullcontext

from ..model.modeling_t5_booru import T5BooruForMaskedLM
from ..vocab import Vocab

from ..iteration import nth, repeatedly
from ..booru_collator import BooruBatchData
  
class SampleSource(Enum):
	Favourite = auto(),
	Sequential = auto(),

@dataclass
class GenerationCallback(TrainerCallback):
	vocab: Vocab
	batches: Iterable[BooruBatchData]
	generation_config: GenerationConfig

	report_to_wandb: bool
	generate_steps: int

	amp_context: torch.cuda.amp.autocast|nullcontext = field(default_factory=nullcontext)

	favourite_batch: BooruBatchData = field(init=False)
	data_it: Iterable[BooruBatchData] = field(init=False)
	batch_source: Iterator[SampleSource] = field(init=False)
	def __post_init__(self):
		it0, it1 = tee(self.batches, 2)
		# self.favourite_batch = nth(it0, 2)
		self.favourite_batch = next(it0)
		# we use repeatedly so that we can circle around once the iterable is exhausted
		self.data_it = repeatedly(it1)
		self.batch_source = iter(cycle((SampleSource.Favourite, SampleSource.Sequential)))

	def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		"""
		Event called at the beginning of a training step. If using gradient accumulation, one training step might take
		several inputs.
		"""
		if state.global_step % self.generate_steps > 0:
			return
		batch_source: SampleSource = next(self.batch_source)

		batch: BooruBatchData = self.favourite_batch if batch_source is SampleSource.Favourite else next(self.data_it)
		# streamer = TokenStreamer(self.vocab)

		# [[self.vocab.tokens[token_ix] for token_ix in caption] for caption in batch['input_ids']]
		# print('\n'.join(''.join('1' if tok else '0' for tok in mask) for mask in batch['attention_mask'].byte()))
		# [[-100 if token_ix == -100 else self.vocab.tokens[token_ix] for token_ix in caption] for caption in batch['labels']]
		# [[self.vocab.tokens[token_ix] for token_ix in caption] for caption in batch['decoder_input_ids']]
		# print('\n'.join(''.join('1' if tok else '0' for tok in mask) for mask in batch['decoder_attention_mask'].byte()))
		# [[self.vocab.tokens[token_ix] for token_ix in caption] for caption in batch['unmasked']]

		# print('\n'.join([' '.join(['-100' if token_ix == -100 else self.vocab.tokens[token_ix] for token_ix in caption[:18]]) for caption in batch['input_ids']]))
		# print('\n'.join([' '.join(['-100' if token_ix == -100 else self.vocab.tokens[token_ix] for token_ix in caption[:18]]) for caption in batch['labels']]))

		with inference_mode(), self.amp_context:
			model: T5BooruForMaskedLM | T5ForConditionalGeneration = kwargs['model']
			model.eval()
			model.gradient_checkpointing_disable()

			# remove accelerate's ConvertOutputsToFp32 wrapper around our model.forward, to ensure autoregressive decoding
			# doesn't receive float32 past_key_values and have to concatenate them with mixed-precision bf16 hidden_states
			wrapped_forward = forward = getattr(model, "forward")
			if '_original_forward' in model.__dict__:
				original_forward = model.__dict__["_original_forward"]
				if original_forward is not None:
					while hasattr(forward, "__wrapped__"):
						forward = forward.__wrapped__
						if forward == original_forward:
							break
					model.forward = types.MethodType(forward, model)

			prediction: LongTensor = model.generate(
				input_ids=batch['input_ids'].to(model.device),
				attention_mask=batch['attention_mask'].to(model.device),
				generation_config=self.generation_config,
				# streamer=streamer,
			)

			# restore accelerate's wrapped forward
			model.forward = wrapped_forward

			model.gradient_checkpointing_enable()
			model.train()

		if self.report_to_wandb:
			import wandb
			metric_key: Literal['prompt_fav', 'prompt_rand'] = 'prompt_fav' if batch_source is SampleSource.Favourite else 'prompt_rand'
			table = wandb.Table(
				data=[
					[
						self.format_caption(input_ids),
						self.format_caption(decoder_input_ids),
						self.format_caption(sample_pred),
					] for input_ids, decoder_input_ids, sample_pred in zip(batch['input_ids'], batch['decoder_input_ids'], prediction)
				],
				columns=['input_ids', 'decoder_input_ids', 'prediction']
			)
			metrics: Dict[Literal['prompt_fav', 'prompt_rand'], wandb.Table] = {
 				metric_key: table,
			}
			wandb.log(metrics, step=state.global_step)
		pass # put breakpoint here

	def	format_token(self, token_ix: int) -> str:
		if token_ix == -100:
			return '-100'
		elif token_ix == self.vocab.token_to_ix[',']:
			return 'CM'
		return self.vocab.tokens[token_ix]

	def	format_caption(self, caption: Iterable[int]) -> str:
		return ' '.join([
			self.format_token(token_ix) for token_ix in caption
		])

	def	format_tokens(self, tokens: LongTensor) -> str:
		return '\n'.join([
			self.format_caption(caption) for caption in tokens
		])
