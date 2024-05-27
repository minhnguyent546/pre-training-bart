"""
A standard transformer model as in Vaswani et al. (2017).
"""

from dataclasses import dataclass
from typing import Literal
import math

from torch import Tensor
from torch import nn
import torch
from torch.nn import functional as F

from . import utils


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
    device: torch.device | Literal['auto'] = 'auto'
    shared_vocab: bool = True
    tie_weights: bool = True  # whether to use tied weights between token embeddings and the pre-softmax linear layer
    hidden_size: int = 512
    num_heads: int = 8
    num_hidden_layers: int = 6
    intermediate_size: int = 2048
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: str = 'relu'

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

class TransformerEmbeddings(nn.Module):
    """Token embeddings and position embeddings, and then layer norm and dropout."""
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_postition_embeddings: int,
        embeddings_dropout: float,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_postition_embeddings, hidden_size)
        self.layer_norm = LayerNormalization(hidden_size)
        self.dropout = nn.Dropout(embeddings_dropout)
        self.register_buffer(
            'position_ids',
            torch.arange(0, max_postition_embeddings),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length = x.shape[:2]
        position_ids = self.position_ids[:seq_length]
        embeddings = self.token_embeddings(x) + self.position_embeddings(position_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
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
        self.activation_fn = utils.get_activation(activation)
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
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            config.hidden_size,
            config.num_heads,
            config.attn_dropout,
        )
        self.feed_forward = FeedForward(
            config.hidden_size,
            config.intermediate_size,
            config.activation,
            config.dropout,
        )
        self.self_attention_norm = LayerNormalization(config.hidden_size)
        self.ff_norm = LayerNormalization(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self.self_attention(x, attn_mask=attn_mask)
        x = self.self_attention_norm(residual + self.dropout(x))

        residual = x
        x = self.feed_forward(x)
        x = self.ff_norm(residual + self.dropout(x))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(
            config.hidden_size,
            config.num_heads,
            config.attn_dropout,
        )
        self.cross_attention = MultiHeadAttention(
            config.hidden_size,
            config.num_heads,
            config.attn_dropout,
        )
        self.feed_forward = FeedForward(
            config.hidden_size,
            config.intermediate_size,
            config.activation,
            config.dropout,
        )
        self.masked_self_attention_norm = LayerNormalization(config.hidden_size)
        self.cross_attention_norm = LayerNormalization(config.hidden_size)
        self.ff_norm = LayerNormalization(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        attn_mask: Tensor | None = None,
        encoder_attn_mask: Tensor | None = None,
    ) -> Tensor:
        residual = x
        x = self.masked_self_attention(x, attn_mask=attn_mask)
        x = self.masked_self_attention_norm(residual + self.dropout(x))

        residual = x
        x = self.cross_attention(x, kv_tensor=encoder_output, attn_mask=encoder_attn_mask)
        x = self.cross_attention_norm(residual + self.dropout(x))

        residual = x
        x = self.feed_forward(x)
        x = self.ff_norm(residual + self.dropout(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        attn_mask: Tensor | None = None,
        encoder_attn_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_output=encoder_output,
                attn_mask=attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )
        return x

class TransformerBase(nn.Module):
    def _tie_weights(self):
        raise NotImplementedError()

    def _init_module_weights_fn(self, module, std: float = 0.02):
        raise NotImplementedError()

    def post_init(self) -> None:
        self._init_model_weights()

    def _init_model_weights(self, std: float = 0.02):
        self.apply(lambda module: self._init_module_weights_fn(module, std=std))

    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

class Transformer(TransformerBase):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.src_embeddings = TransformerEmbeddings(
            config.src_vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            config.dropout,
        )
        if not self.config.shared_vocab and config.target_vocab_size is None:
            raise ValueError('`target_vocab_size` must be provided if `shared_vocab` is `False`')
        self.target_embeddings = TransformerEmbeddings(
            config.target_vocab_size or config.src_vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            config.dropout,
        )
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

        if self.config.tie_weights:
            self._tie_weights()
        self.post_init()

    @property
    def src_token_embeddings(self):
        return self.src_embeddings.token_embeddings

    @property
    def target_token_embeddings(self):
        return self.target_embeddings.token_embeddings

    def _tie_weights(self) -> None:
        if self.target_token_embeddings.weight.shape == self.src_token_embeddings.weight.shape:
            self.target_token_embeddings.weight = self.src_token_embeddings.weight

    def _init_module_weights_fn(self, module, std: float = 0.02):
        """Following the same initialization as in BERT."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight.data[module.padding_idx])

    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attn_mask: Tensor | None = None,
        decoder_attn_mask: Tensor | None = None
    ) -> Tensor:
        encoder_input = self.src_embeddings(input_ids)
        if attn_mask is None:
            attn_mask = input_ids != self.config.src_pad_token_id
        if attn_mask.dim() == 2:
            attn_mask = utils.create_encoder_4d_attn_mask(input_ids, attn_mask)
        encoder_output = self.encoder(encoder_input, attn_mask=attn_mask)

        if decoder_attn_mask is None:
            decoder_attn_mask = decoder_input_ids != self.config.target_pad_token_id
        if decoder_attn_mask.dim() == 2:
            decoder_attn_mask = utils.create_decoder_4d_attn_causal_mask(decoder_input_ids, decoder_attn_mask)
        decoder_input = self.target_embeddings(decoder_input_ids)
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            attn_mask=decoder_attn_mask,
            encoder_attn_mask=attn_mask,
        )
        return decoder_output

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
