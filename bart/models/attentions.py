import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttnImpl(str, Enum):
    DOT_PRODUCT = 'dot_product'
    SCALED_DOT_PRODUCT = 'scaled_dot_product'
    SLIDING_WINDOW = 'sliding_window'
    ADDITIVE = 'additive'

    @classmethod
    def list(cls):
        return list(map(lambda element: element.value, cls))

def dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: bool = True,
    attn_mask: Tensor | None = None,
    attn_dropout: float | nn.Dropout | None = None,
) -> Tensor:

    batch_size, _, query_length, head_dim = query.shape
    key_length = key.shape[2]
    if attn_mask is not None:
        expected_mask_shape = (batch_size, 1, query_length, key_length)
        if tuple(attn_mask.shape) != expected_mask_shape:
            raise ValueError(
                f'Expected attn_mask has shape of {expected_mask_shape}, '
                f'got {tuple(attn_mask.shape)}'
            )
    if scale == False:
        # in case it is not scaled dot-product attention, we will scale query instead of query @ key^T
        query = query / math.sqrt(head_dim)
    # query: (batch_size, num_heads, query_length, head_dim)
    # key: (batch_size, num_heads, key_length, head_dim)
    # value: (batch_size, num_heads, key_length, head_dim)
    # attention_probs: (batch_size, num_heads, query_length, key_length)
    attn_probs = (query @ key.transpose(-2, -1))
    if scale == True:
        attn_probs = attn_probs / math.sqrt(head_dim)
    if attn_mask is not None:
        attn_probs.masked_fill_(attn_mask == False, float('-inf'))

    attn_probs = F.softmax(attn_probs, dim=-1)
    if attn_dropout is not None:
        if isinstance(attn_dropout, float):
            attn_dropout = nn.Dropout(attn_dropout)
        attn_probs = attn_dropout(attn_probs)

    output = attn_probs @ value  # (batch_size, num_heads, query_length, head_dim)
    return output

def loop_sliding_window_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_window_size: int,
    attn_mask: Tensor | None = None,
    attn_dropout: float | nn.Dropout | None = None,
) -> Tensor:
    batch_size, num_heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    assert query_length == key_length
    if attn_mask is not None:
        expected_mask_shape = (batch_size, 1, query_length, key_length)
        if tuple(attn_mask.shape) != expected_mask_shape:
            raise ValueError(
                f'Expected attn_mask has shape of {expected_mask_shape}, '
                f'got {tuple(attn_mask.shape)}'
            )
    head_dim = query.size(-1)
    # query: (batch_size, num_heads, query_length, head_dim)
    # key: (batch_size, num_heads, key_length, head_dim)
    # value: (batch_size, num_heads, key_length, head_dim)

    if attn_dropout is not None:
        if isinstance(attn_dropout, float):
            attn_dropout = nn.Dropout(attn_dropout)

    assert attn_window_size % 2 != 0
    half_window_size = (attn_window_size - 1) // 2
    # we will compute attention probabilities for each query
    output = torch.empty((batch_size, num_heads, query_length, head_dim), dtype=query.dtype, device=query.device)
    for query_idx in range(query_length):
        start_pos = max(0, query_idx - half_window_size)
        end_pos = min(query_length - 1, query_idx + half_window_size)
        current_query = query[..., [query_idx], :]  # (batch_size, num_heads, 1, head_dim)
        local_key = key[..., start_pos:end_pos, :]  # (batch_size, num_heads, window_size, head_dim)
        local_value = value[..., start_pos:end_pos, :]  # (batch_size, num_heads, window_size, head_dim)
        local_attn_probs = current_query @ local_key.transpose(-2, -1) / math.sqrt(head_dim)
        if attn_mask is not None:
            local_attn_mask = attn_mask[..., [query_idx], start_pos:end_pos, :]
            local_attn_probs.masked_fill_(local_attn_mask == False, float('-inf'))

        local_attn_probs = F.softmax(local_attn_probs, dim=-1)
        if attn_dropout is not None:
            local_attn_probs = attn_dropout(local_attn_probs)
        current_output = local_attn_probs @ local_value  # (batch_size, num_heads, 1, head_dim)
        output[..., [query_idx], :] = current_output

    return output

def additive_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    w_a: nn.Linear,
    attn_mask: Tensor | None = None,
    attn_dropout: float | nn.Dropout | None = None,
) -> Tensor:
    batch_size, _, query_length, _ = query.shape
    key_length = key.shape[2]
    if attn_mask is not None:
        expected_mask_shape = (batch_size, 1, query_length, key_length)
        if tuple(attn_mask.shape) != expected_mask_shape:
            raise ValueError(
                f'Expected attn_mask has shape of {expected_mask_shape}, '
                f'got {tuple(attn_mask.shape)}'
            )
    # before:
    # query: (batch_size, num_heads, query_length, head_dim)
    # key: (batch_size, num_heads, key_length, head_dim)
    #
    # after:
    # query: (batch_size, num_heads, query_length, 1, head_dim)
    # key: (batch_size, num_heads, 1, key_length, head_dim)
    query = query.unsqueeze(3)  # (batch_size, num_heads)
    key = key.unsqueeze(2)
    attn_probs = w_a(F.tanh(query + key))  # (batch_size, num_heads, query_length, key_length, 1)
    attn_probs = attn_probs.unsqueeze(-1)  # (batch_size, num_heads, query_length, key_length)
    if attn_mask is not None:
        attn_probs.masked_fill_(attn_mask == False, float('-inf'))

    attn_probs = F.softmax(attn_probs, dim=-1)
    if attn_dropout is not None:
        if isinstance(attn_dropout, float):
            attn_dropout = nn.Dropout(attn_dropout)
        attn_probs = attn_dropout(attn_probs)

    # value: (batch_size, num_heads, key_length, head_dim)
    output = attn_probs @ value  # (batch_size, num_heads, query_length, head_dim)
    return output
