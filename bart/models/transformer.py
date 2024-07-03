"""
A standard transformer model as in Vaswani et al. (2017).
"""

import math
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor
from torch import nn

import bart.models.utils as model_utils
from bart.models.attentions import (
    AttnImpl,
    additive_attention,
    dot_product_attention,
    loop_sliding_window_attention,
)
from bart.models.utils import initialize_bert_params_fn


KvCacheType: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]

@dataclass
class TransformerConfig:
    src_pad_token_id: int
    target_pad_token_id: int
    target_start_token_id: int
    target_end_token_id: int
    src_vocab_size: int = 32000
    target_vocab_size: int | None = None  # `src_vocab_size` will be used as common vocab size if `shared_vocab` = `True`
    src_seq_length: int = 128
    target_seq_length: int = 128
    max_position_embeddings: int = 512  # just in case we need to train with larger sequence length at some point
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared_vocab: bool = True
    tie_weights: bool = True  # whether to use tied weights between token embeddings and the pre-softmax linear layer
    hidden_size: int = 512
    intermediate_size: int = 512 * 4
    encoder_num_heads: int = 8
    encoder_num_kv_heads: int | None = None  # it should be `None` or equal to `encoder_num_heads` as it does not effect too much in training time
    encoder_num_hidden_layers: int = 6
    encoder_attn_impl: str = 'scaled_dot_product'
    encoder_attn_window_size: int | None = None
    decoder_num_heads: int = 8
    decoder_num_kv_heads: int | None = None
    decoder_num_hidden_layers: int = 6
    decoder_attn_impl: str = 'scaled_dot_product'
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: str = 'gelu'
    pre_norm: bool = False  # whether to place LayerNorm before each sub-layer (also known as pre-norm)
    init_std: float = 0.02

@dataclass
class TransformerOutput:
    encoder_output: Tensor
    decoder_output: Tensor | None = None
    kv_caches: tuple[KvCacheType, ...] | None = None

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-7):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.norm_eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.norm_eps).sqrt()
        y = (x - mean) / std
        y = self.weight * y + self.bias
        return y

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, scale_factor: float = 1.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.scale_factor = scale_factor

    @property
    def weight(self) -> Tensor:
        return self.token_embeddings.weight

    def forward(self, x: Tensor) -> Tensor:
        embeddings = self.token_embeddings(x) * self.scale_factor
        return embeddings

class PositionEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, max_postition_embeddings: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_postition_embeddings, hidden_size)

    def forward(self, x: Tensor, cached_kv_length: int = 0) -> Tensor:
        # when `cached_kv_length` > 0, seq_length most likely will be 1
        batch_size, seq_length = x.shape[:2]
        positions = torch.arange(
            cached_kv_length,
            cached_kv_length + seq_length,
            device=x.device,
            dtype=torch.int32,
        ).expand(batch_size, -1)  # (batch_size, seq_length)
        embeddings = self.position_embeddings(positions)  # (batch_size, seq_length, hidden_size)
        return embeddings

class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.proj = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = model_utils.get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    A multi-head attention module which supports various types of
    attention mechanisms and grouped-query attention.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float,
        attn_impl: str | AttnImpl = AttnImpl.SCALED_DOT_PRODUCT,
        num_kv_heads: int | None = None,
        is_decoder: bool = False,
        attn_window_size: int | None = None
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f'The hidden size {hidden_size} is not divisible by '
                f'the number of attention heads {num_heads}'
            )
        if num_kv_heads is not None and num_heads % num_kv_heads != 0:
            raise ValueError('Expected `num_heads` is divisible by `num_kv_heads`')
        if attn_impl not in AttnImpl:
            raise ValueError(
                f'Unknown AttnType value: {attn_impl}. '
                f'Possible values are: {', '.join(AttnImpl.list())}')
        if attn_impl == AttnImpl.SLIDING_WINDOW:
            if attn_window_size is None:
                raise ValueError('When using sliding window attention, `attn_window_size` must be provided')
            assert is_decoder == False
            if attn_window_size % 2 == 0 or attn_window_size <= 0:
                raise ValueError('Please provide a positive odd number `attn_window_size`')

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn_impl = attn_impl
        self.attn_dropout = attn_dropout
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.is_decoder = is_decoder
        self.attn_window_size = attn_window_size

        self.head_dim = hidden_size // num_heads
        if self.attn_impl == AttnImpl.ADDITIVE:
            self.w_a = nn.Linear(self.hidden_size, 1)
        self.num_kv_replicas = self.num_heads // self.num_kv_heads

        self.w_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.w_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.w_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.w_o = nn.Linear(self.num_heads * self.head_dim, hidden_size)

    def forward(
        self,
        x: Tensor,
        kv_states: Tensor | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """if `kv_states` is passed, then it will used as key and value tensors (cross attention)"""
        batch_size = x.shape[0]
        is_cross_attention = kv_states is not None

        query = self._reshape_for_mha(self.w_q(x), batch_size, self.num_heads)
        if is_cross_attention and kv_cache is not None:
            # cross attention with cache
            # check if seq_lengths are equal
            assert kv_cache[0].shape[2] == kv_states.shape[1]
            key, value = kv_cache
        elif is_cross_attention:
            # cross-attention
            key = self._reshape_for_mha(self.w_k(kv_states), batch_size, self.num_kv_heads)
            value = self._reshape_for_mha(self.w_v(kv_states), batch_size, self.num_kv_heads)
        elif kv_cache is not None:
            # self-attention with cache
            cur_key = self._reshape_for_mha(self.w_k(x), batch_size, self.num_kv_heads)
            cur_value = self._reshape_for_mha(self.w_v(x), batch_size, self.num_kv_heads)
            key = torch.cat([kv_cache[0], cur_key], dim=2)
            value = torch.cat([kv_cache[1], cur_value], dim=2)
        else:
            # self-attention
            key = self._reshape_for_mha(self.w_k(x), batch_size, self.num_kv_heads)
            value = self._reshape_for_mha(self.w_v(x), batch_size, self.num_kv_heads)

        if self.is_decoder:
            # kv_cache is always `None` in encoder
            kv_cache = (key, value)

        key = self._repeat_kv(key)
        value = self._repeat_kv(value)

        # attention mechanisms
        if self.attn_impl == AttnImpl.DOT_PRODUCT:
            output = dot_product_attention(
                query, key, value, scale=False,
                attn_mask=attn_mask, attn_dropout=self.attn_dropout,
            )
        elif self.attn_impl == AttnImpl.SCALED_DOT_PRODUCT:
            output = dot_product_attention(
                query, key, value, scale=True,
                attn_mask=attn_mask, attn_dropout=self.attn_dropout,
            )
        elif self.attn_impl == AttnImpl.SLIDING_WINDOW:
            output = loop_sliding_window_attention(
                query, key, value, attn_window_size=self.attn_window_size,
                attn_mask=attn_mask, attn_dropout=self.attn_dropout,
            )
        elif self.attn_impl == AttnImpl.ADDITIVE:
            output = additive_attention(
                query, key, value, self.w_a,
                attn_mask=attn_mask, attn_dropout=self.attn_dropout,
            )
        else:
            raise ValueError(f'Expected attn_type is an AttnType, but got {type(self.attn_impl)}')

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.w_o(output)
        return output, kv_cache

    def _reshape_for_mha(
        self,
        x: Tensor,
        batch_size: int,
        num_heads: int,
        seq_length: int = -1,
    ) -> Tensor:
        return x.view(batch_size, seq_length, num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_replicas <= 1:
            return x

        # x:  (batch_size, num_kv_heads, seq_length, head_dim)
        batch_size, num_kv_heads, seq_length, head_dim = x.shape
        x = x.unsqueeze(2)  # (batch_size, num_kv_heads, 1, seq_length, head_dim)
        x = x.expand(batch_size, num_kv_heads, self.num_replicas, seq_length, head_dim)
        x = x.reshape(batch_size, num_kv_heads * self.num_replicas, seq_length, head_dim)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        activation: str,
        attn_impl: str,
        num_kv_heads: int | None = None,
        pre_norm: bool = False,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        attn_window_size: int | None = None,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_dropout,
            attn_impl=attn_impl,
            num_kv_heads=num_kv_heads,
            attn_window_size=attn_window_size,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            activation,
            dropout,
        )
        self.pre_norm = pre_norm
        self.layer_norms = nn.ModuleList([
            LayerNormalization(hidden_size)
            for _ in range(2)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self._maybe_layer_norm(x, 0, is_before=True)
        x, _ = self.self_attention(x, attn_mask=attn_mask)
        x = residual + self.dropout(x)
        x = self._maybe_layer_norm(x, 0)

        residual = x
        x = self._maybe_layer_norm(x, 1, is_before=True)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        x = self._maybe_layer_norm(x, 1)
        return x

    def _maybe_layer_norm(
        self,
        x: Tensor,
        sub_layer_idx: int,
        is_before: bool = False,
    ) -> Tensor:
        assert sub_layer_idx < len(self.layer_norms)
        if not (is_before ^ self.pre_norm):  # is_before === pre_norm
            x = self.layer_norms[sub_layer_idx](x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        activation: str,
        attn_impl: str,
        num_kv_heads: int | None = None,
        pre_norm: bool = False,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_dropout,
            attn_impl=attn_impl,
            num_kv_heads=num_kv_heads,
            is_decoder=True,
        )
        self.cross_attention = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_dropout,
            attn_impl=attn_impl,
            num_kv_heads=num_kv_heads,
            is_decoder=True,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            activation,
            dropout,
        )
        self.pre_norm = pre_norm
        self.layer_norms = nn.ModuleList([
            LayerNormalization(hidden_size)
            for _ in range(3)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        attn_mask: Tensor | None = None,
        encoder_attn_mask: Tensor | None = None,
        kv_cache: KvCacheType | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        self_attn_kv_cache = kv_cache[:2] if kv_cache is not None else None
        residual = x
        x = self._maybe_layer_norm(x, 0, is_before=True)
        x, new_self_attn_kv_cache = self.masked_self_attention(
            x,
            kv_cache=self_attn_kv_cache,
            attn_mask=attn_mask,
        )
        x = residual + self.dropout(x)
        x = self._maybe_layer_norm(x, 0)

        cross_attn_kv_cache = kv_cache[-2:] if kv_cache is not None else None
        residual = x
        x = self._maybe_layer_norm(x, 1, is_before=True)
        x, new_cross_attn_kv_cache = self.cross_attention(
            x,
            kv_states=encoder_output,
            kv_cache=cross_attn_kv_cache,
            attn_mask=encoder_attn_mask,
        )
        x = residual + self.dropout(x)
        x = self._maybe_layer_norm(x, 1)

        residual = x
        x = self._maybe_layer_norm(x, 2, is_before=True)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        x = self._maybe_layer_norm(x, 2)

        new_kv_cache = None
        if use_cache:
            assert new_self_attn_kv_cache is not None
            assert new_cross_attn_kv_cache is not None
            new_kv_cache = new_self_attn_kv_cache + new_cross_attn_kv_cache
        return x, new_kv_cache

    def _maybe_layer_norm(
        self,
        x: Tensor,
        sub_layer_idx: int,
        is_before: bool = False,
    ) -> Tensor:
        assert sub_layer_idx < len(self.layer_norms)
        if not (is_before ^ self.pre_norm):  # is_before === pre_norm
            x = self.layer_norms[sub_layer_idx](x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embeddings = InputEmbeddings(
            config.src_vocab_size,
            config.hidden_size,
            scale_factor=math.sqrt(config.hidden_size),
        )
        self.position_embeddings = PositionEmbeddings(config.hidden_size, config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.hidden_size,
                config.encoder_num_heads,
                config.intermediate_size,
                config.activation,
                config.encoder_attn_impl,
                num_kv_heads=config.encoder_num_kv_heads,
                pre_norm=config.pre_norm,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                attn_window_size=config.encoder_attn_window_size,
            )
            for _ in range(config.encoder_num_hidden_layers)
        ])
        self.pre_norm = config.pre_norm
        if self.pre_norm:
            self.layer_norm = LayerNormalization(config.hidden_size)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, embed_tokens: bool = True) -> Tensor:
        if attn_mask is not None:
            attn_mask = model_utils.create_encoder_4d_attn_mask(attn_mask, x.shape[:2])

        # embed tokens and positions
        if embed_tokens:
            assert hasattr(self, 'token_embeddings')
            x = self.token_embeddings(x)
        else:
            assert x.dim() == 3
        x = x + self.position_embeddings(x)
        x = self.dropout(x)

        # pass through encoder layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # additional layer norm if we use pre-norm
        if self.pre_norm:
            x = self.layer_norm(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        if not config.shared_vocab and config.target_vocab_size is None:
            raise ValueError('`target_vocab_size` must be provided if `shared_vocab` is `False`')
        self.token_embeddings = InputEmbeddings(
            config.target_vocab_size or config.src_vocab_size,
            config.hidden_size,
            scale_factor=math.sqrt(config.hidden_size),
        )
        self.position_embeddings = PositionEmbeddings(config.hidden_size, config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.hidden_size,
                config.decoder_num_heads,
                config.intermediate_size,
                config.activation,
                config.decoder_attn_impl,
                num_kv_heads=config.decoder_num_kv_heads,
                pre_norm=config.pre_norm,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
            )
            for _ in range(config.decoder_num_hidden_layers)
        ])
        self.pre_norm = config.pre_norm
        if self.pre_norm:
            self.layer_norm = LayerNormalization(config.hidden_size)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        attn_mask: Tensor | None = None,
        encoder_attn_mask: Tensor | None = None,
        kv_caches: tuple[KvCacheType, ...] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[KvCacheType] | None]:
        cached_kv_length = kv_caches[0][0].shape[2] if kv_caches is not None else 0

        attn_mask = model_utils.create_decoder_4d_causal_attn_mask(
            attn_mask,
            x.shape,
            cached_kv_length=cached_kv_length,
            device=x.device,
        )
        if encoder_attn_mask is not None:
            encoder_attn_mask = model_utils.create_encoder_4d_attn_mask(
                encoder_attn_mask,
                encoder_output.shape[:2],
                target_seq_length=x.shape[1],
            )

        # embed tokens and positions
        x = self.token_embeddings(x) + self.position_embeddings(x, cached_kv_length=cached_kv_length)
        x = self.dropout(x)

        # pass through decoder layers
        next_kv_caches = () if use_cache else None
        for idx, layer in enumerate(self.layers):
            kv_cache = kv_caches[idx] if kv_caches is not None else None
            x, new_kv_cache = layer(
                x,
                encoder_output=encoder_output,
                attn_mask=attn_mask,
                encoder_attn_mask=encoder_attn_mask,
                kv_cache=kv_cache,
                use_cache=use_cache,
            )
            if use_cache:
                assert new_kv_cache is not None
                next_kv_caches += (new_kv_cache,)

        # additional layer norm if we use pre-norm
        if self.pre_norm:
            x = self.layer_norm(x)
        return x, next_kv_caches

class TransformerBase(nn.Module):
    def _tie_weights(self):
        raise NotImplementedError()

    def _init_model_weights(self, std: float = 0.02):
        raise NotImplementedError()

    def post_init(self, std: float = 0.02) -> None:
        self._init_model_weights(std=std)

    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

class Transformer(TransformerBase):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

        if self.config.tie_weights:
            self._tie_weights()
        self.post_init(std=self.config.init_std)

    def _tie_weights(self) -> None:
        if not self.config.shared_vocab:
            return

        assert self.encoder.token_embeddings.weight.shape == self.decoder.token_embeddings.weight.shape
        self.encoder.token_embeddings = self.decoder.token_embeddings

    def _init_model_weights(self, std: float = 0.02) -> None:
        self.apply(lambda module: initialize_bert_params_fn(module, std=std))

    def forward(
        self,
        encoder_input_ids: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        encoder_attn_mask: Tensor | None = None,
        decoder_attn_mask: Tensor | None = None,
        encoder_output: Tensor | None = None,
        kv_caches: tuple[KvCacheType, ...] | None = None,
        use_cache: bool = False,
    ) -> TransformerOutput:
        if self.training:
            use_cache = False
        if encoder_output is None:
            if encoder_input_ids is None:
                raise ValueError('If `encoder_output` is not passed, `encoder_input_ids` can not be `None`')
            encoder_output = self.encoder(encoder_input_ids, attn_mask=encoder_attn_mask)

        assert encoder_output is not None
        decoder_output = None
        new_kv_caches = None
        if decoder_input_ids is not None:
            decoder_output, new_kv_caches = self.decoder(
                decoder_input_ids,
                encoder_output=encoder_output,
                attn_mask=decoder_attn_mask,
                encoder_attn_mask=encoder_attn_mask,
                kv_caches=kv_caches,
                use_cache=use_cache,
            )
        return TransformerOutput(
            encoder_output=encoder_output,
            decoder_output=decoder_output,
            kv_caches=new_kv_caches,
        )
