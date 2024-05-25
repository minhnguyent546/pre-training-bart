import torch
from torch import nn


def get_activation(act_type: str) -> nn.Module:
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(
            f'Unrecognized activation type: {act_type}. '
            'Possible values are "relu", "gelu".'
        )

def get_device(device: torch.device | str = 'auto') -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)
