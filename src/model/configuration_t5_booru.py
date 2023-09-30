from transformers.configuration_utils import PretrainedConfig
from ..ceil_to_multiple import ceil_to_multiple
from typing import TypedDict, Optional

class SReparamConfig(TypedDict):
    n_iters: int
    n_iters_init: int
    eps: float
    learn_gamma: bool
    register_v_during_construction: bool

class AttnKeyLenCompensation(TypedDict):
    avg_key_len_init: int
    population_init: int
    population_max: int

class T5BooruConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`T5Model`] or a [`TFT5Model`]. It is used to
    instantiate a T5 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the T5
    [t5-small](https://huggingface.co/t5-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Model`] or [`TFT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
            be defined as `num_heads * d_kv`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `T5Block`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the
            `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """
    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    s_reparam_config: SReparamConfig
    prompt_len_avg: AttnKeyLenCompensation
    continuation_len_avg: AttnKeyLenCompensation

    def __init__(
        self,
        vocab_size_nominal=32169,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        decoder_conv_in=True,
        decoder_mlp=False,
        decoder_start_token_id=0,
        encoder_conv_in=True,
        max_ctx_len=256,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        label_ignore_index=-100,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        # measured average length of input_ids over entire Danbooru corpus
        prompt_len_avg: Optional[AttnKeyLenCompensation] = AttnKeyLenCompensation(
            avg_key_len_init=81,
            population_init=5_626_898,
            population_max=2**31,
        ),
        # measured average length of labels over first batch of Danbooru corpus
        continuation_len_avg: Optional[AttnKeyLenCompensation] = AttnKeyLenCompensation(
            avg_key_len_init=18,
            population_init=256,
            population_max=2**31,
        ),
        # reduces t5-small's params from 65,997,184 -> 54,983,552
        # TODO: grow d_ff
        tie_conv_in=True,
        tie_encoder_ffns=True,
        tie_word_embeddings=False,
        sreparam_multi_head=False,
        use_sigma_reparam=False,
        use_attn_pre_ln=False,
        use_attn_post_ln=True,
        fix_lm_head_weight_init=True,
        use_cache=True,
        pad_vocab_to_multiple=8,
        pad_token_id=0,
        eos_token_id=1,
        s_reparam_config=SReparamConfig(
            n_iters=1,
            n_iters_init=15,
            eps=1e-12,
            learn_gamma=True,
            register_v_during_construction=True,
        ),
        **kwargs,
    ):
        self.vocab_size_nominal = vocab_size_nominal
        self.pad_vocab_to_multiple = pad_vocab_to_multiple
        self.vocab_size = ceil_to_multiple(vocab_size_nominal, pad_vocab_to_multiple)
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.decoder_conv_in = decoder_conv_in
        self.decoder_mlp = decoder_mlp
        self.encoder_conv_in = encoder_conv_in
        self.max_ctx_len = max_ctx_len
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.label_ignore_index = label_ignore_index
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.tie_conv_in = tie_conv_in
        self.tie_encoder_ffns = tie_encoder_ffns
        self.use_sigma_reparam = use_sigma_reparam
        self.sreparam_multi_head = sreparam_multi_head
        self.s_reparam_config = s_reparam_config
        self.prompt_len_avg = prompt_len_avg
        self.continuation_len_avg = continuation_len_avg
        self.use_attn_pre_ln = use_attn_pre_ln
        self.use_attn_post_ln = use_attn_post_ln
        self.fix_lm_head_weight_init = fix_lm_head_weight_init
        self.use_cache = use_cache

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        if use_sigma_reparam:
            assert not tie_word_embeddings, 'tie_word_embeddings is not supported when lm_head is sigma-reparameterised.'

        super().__init__(
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )