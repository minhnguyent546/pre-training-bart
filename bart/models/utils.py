from torch import Tensor, nn
import torch


def get_activation(act_type: str) -> nn.Module:
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Unrecognized activation type: {act_type}.')

def get_device(device: torch.device | str = 'auto') -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def shift_tokens_right(tokens: Tensor, start_token_id: int) -> Tensor:
    if tokens.dim() != 2:
        raise ValueError(f'Expected tokens tensor is a 2D tensor, but got a tensor with shape: {tokens.size()}')

    shifted = tokens.new_zeros(tokens.shape)
    shifted[:, :1] = start_token_id
    shifted[:, 1:] = tokens[:, :-1].detach().clone()
    return shifted

def create_encoder_4d_attn_mask(input_ids: Tensor, attn_mask: Tensor) -> Tensor:
    if attn_mask.shape != input_ids.shape:
        raise ValueError(
            'Expected input_ids and attn_mask have the same shape, '
            f'got {input_ids.shape} and {attn_mask.shape}'
        )
    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
    return attn_mask

def create_decoder_4d_attn_causal_mask(
    decoder_input_ids: Tensor,
    decoder_attn_mask: Tensor,
) -> Tensor:
    if decoder_attn_mask.shape != decoder_input_ids.shape:
        raise ValueError(
            'Expected decoder_input_ids and decoder_attn_mask have the same shape, '
            f'got {decoder_input_ids.shape} and {decoder_attn_mask.shape}'
        )
    seq_length = decoder_input_ids.shape[1]
    causal_mask = torch.tril(torch.ones(seq_length, seq_length)).bool().to(decoder_attn_mask.device)
    decoder_attn_mask = decoder_attn_mask.unsqueeze(1).unsqueeze(2)
    decoder_attn_causal_mask = decoder_attn_mask & causal_mask
    return decoder_attn_causal_mask
