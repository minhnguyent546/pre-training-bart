"""
A standard transformer model as in Vaswani et al. (2017).
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

import bart.models.utils as model_utils
from bart.models.utils import initialize_bert_params_fn


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
    encoder_num_hidden_layers: int = 6
    decoder_num_heads: int = 8
    decoder_num_hidden_layers: int = 6
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: str = 'gelu'
    pre_norm: bool = False  # whether to place LayerNorm before each sub-layer (also known as pre-norm)
    init_std: float = 0.02

@dataclass
class TransformerOutput:
    encoder_output: Tensor
    decoder_output: Tensor | None = None

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
        self.register_buffer(
            'position_ids',
            torch.arange(0, max_postition_embeddings),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length = x.shape[:2]
        position_ids = self.position_ids[:seq_length]
        embeddings = self.position_embeddings(position_ids)
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
    """A standard scaled dot-product attention."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float,
    ):
        super().__init__()
        if not hidden_size % num_heads == 0:
            raise ValueError(
                f'The hidden size {hidden_size} is not divisible by '
                f'the number of attention heads {num_heads}'
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, x: Tensor,
        kv_tensor: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """if `kv_tensor` is not `None`, then it is used as key and value tensors (cross attention)"""
        batch_size, seq_length = x.shape[:2]
        is_cross_attention = kv_tensor is not None

        query = self.w_q(x)
        if is_cross_attention:
            # cross-attention
            key = self.w_k(kv_tensor)
            value = self.w_v(kv_tensor)
        else:
            # self-attention
            key = self.w_k(x)
            value = self.w_v(x)

        # query, key, value: (batch_size, seq_length, hidden_size) -> (batch_size, num_heads, seq_length, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        y = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            attn_dropout=self.attn_dropout,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        y = self.w_o(y)
        return y

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        activation: str,
        pre_norm: bool = False,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_dropout,
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
        x = self.self_attention(x, attn_mask=attn_mask)
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
        pre_norm: bool = False,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_dropout,
        )
        self.cross_attention = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_dropout,
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
    ) -> Tensor:
        residual = x
        x = self._maybe_layer_norm(x, 0, is_before=True)
        x = self.masked_self_attention(x, attn_mask=attn_mask)
        x = residual + self.dropout(x)
        x = self._maybe_layer_norm(x, 0)

        residual = x
        x = self._maybe_layer_norm(x, 1, is_before=True)
        x = self.cross_attention(x, kv_tensor=encoder_output, attn_mask=encoder_attn_mask)
        x = residual + self.dropout(x)
        x = self._maybe_layer_norm(x, 1)

        residual = x
        x = self._maybe_layer_norm(x, 2, is_before=True)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        x = self._maybe_layer_norm(x, 2)
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
                pre_norm=config.pre_norm,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
            )
            for _ in range(config.encoder_num_hidden_layers)
        ])
        self.pre_norm = config.pre_norm
        if self.pre_norm:
            self.layer_norm = LayerNormalization(config.hidden_size)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, embed_tokens: bool = True) -> Tensor:
        # embed tokens and positions
        if embed_tokens:
            assert hasattr(self, 'token_embeddings')
            x = self.token_embeddings(x)
        x = x + self.position_embeddings(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
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
    ) -> Tensor:
        # embed tokens and positions
        x = self.token_embeddings(x) + self.position_embeddings(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(
                x,
                encoder_output=encoder_output,
                attn_mask=attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )
        if self.pre_norm:
            x = self.layer_norm(x)
        return x

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
    ) -> TransformerOutput:
        if encoder_output is None:
            if encoder_input_ids is None:
                raise ValueError('If `encoder_output` is not passed, `encoder_input_ids` can not be `None`')
            if encoder_attn_mask is None:
                encoder_attn_mask = encoder_input_ids != self.config.src_pad_token_id
            if encoder_attn_mask.dim() == 2:
                encoder_attn_mask = model_utils.create_encoder_4d_attn_mask(encoder_input_ids, encoder_attn_mask)
            encoder_output = self.encoder(encoder_input_ids, attn_mask=encoder_attn_mask)

        assert encoder_output is not None
        decoder_output = None
        if decoder_input_ids is not None:
            if decoder_attn_mask is None:
                decoder_attn_mask = decoder_input_ids != self.config.target_pad_token_id
            if decoder_attn_mask.dim() == 2:
                decoder_attn_mask = model_utils.create_decoder_4d_attn_causal_mask(decoder_input_ids, decoder_attn_mask)
            decoder_output = self.decoder(
                decoder_input_ids,
                encoder_output=encoder_output,
                attn_mask=decoder_attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )
        return TransformerOutput(
            encoder_output=encoder_output,
            decoder_output=decoder_output,
        )

def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    attn_dropout: float | nn.Dropout | None = None,
) -> Tensor:
    if attn_mask is not None and attn_mask.dim() != 4:
        raise ValueError(f'Expected attn_mask is a 4D tensor, got a tensor with shape: {attn_mask.size()}')

    d_k = query.size(-1)
    attention_probs = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        attention_probs.masked_fill_(attn_mask == False, float('-inf'))

    attention_probs = F.softmax(attention_probs, dim=-1)
    if attn_dropout is not None:
        if isinstance(attn_dropout, float):
            attn_dropout = nn.Dropout(attn_dropout)
        attention_probs = attn_dropout(attention_probs)

    output = attention_probs @ value
    return output
