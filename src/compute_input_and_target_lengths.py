from typing import NamedTuple

class InputAndOutputLength(NamedTuple):
    input_length: int
    output_length: int

class InputAndTargetLengths(NamedTuple):
    tokens_length: int
    targets_length: int

def compute_input_and_target_lengths(
    inputs_length: int,
    noise_density: float,
    mean_noise_span_length: float,
) -> InputAndTargetLengths:
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length: int) -> InputAndOutputLength:
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens: int = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length: int = num_nonnoise_tokens + num_noise_spans + 1
        _output_length: int = num_noise_tokens + num_noise_spans + 1
        return InputAndOutputLength(
            input_length=_input_length,
            output_length=_output_length,
        )

    tokens_length: int = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return InputAndTargetLengths(
        tokens_length=tokens_length,
        targets_length=targets_length,
    )