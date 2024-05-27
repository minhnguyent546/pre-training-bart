import os
import random
import re
import sys
import yaml

import emoji

import numpy as np

from torch import nn
import torch

from models import BartBase


url_regex = re.compile(r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))")
NON_PRINTABLE_TABLE = {
    i: None for i in range(0, sys.maxunicode + 1)
    if not chr(i).isprintable()
}

def set_random_seed(seed: int = 0x3f3f3f3f):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml_config(config_path: str):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

def chunks(data: list | str, chunk_size: int = 1_000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

def noam_decay(step_num: int, d_model: int = 768, warmup_steps: int = 4000):
    """
    As described in https://arxiv.org/pdf/1706.03762.pdf
    """
    step_num = max(step_num, 1)
    return d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def wnext(low: int, high: int, time: int = 0) -> int:
    """Uneven distribution

    If `time` = 0: this is equivalent to `np.random.randint(low, high)`
    If `time` > 0: `wnext` is defined as `max(np.random.randint(low, high, size=(time + 1,)))`
    If `time` < 0: this case is same as `time` > 0 but take min instead of max.
    """

    values = np.random.randint(low, high, size=(abs(time) + 1),)
    if time == 0:
        return values
    elif time > 0:
        return np.max(values)
    else:
        return np.min(values)

def make_optimizer(
    model: BartBase,
    optim_type: str,
    learning_rate: float,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
    exclude_module_list: tuple[nn.Module, ...] = (),
) -> torch.optim.Optimizer:
    decay_params = get_param_names(model, exclude_module_list)
    # also exclude biases
    decay_params = [name for name in decay_params if not name.endswith('bias')]

    param_groups = [
        {
            'params': [param for name, param in model.named_parameters() if name in decay_params],
            'weight_decay': weight_decay,
        },
        {
            'params': [param for name, param in model.named_parameters() if name not in decay_params],
            'weight_decay': 0.0,
        },
    ]
    optim_type = optim_type.lower()
    if optim_type == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
        )
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
        )
    else:
        raise ValueError(f'Unsupported optimizer type: {optim_type}. Possible values are: adam, adamw')

    return optimizer

def get_param_names(
    module: nn.Module,
    exclude_module_list: tuple[nn.Module, ...] = (),
) -> list[str]:
    """Get parameter names but exclude those in `exclude_module_list`."""
    param_names = []
    for child_name, child in module.named_children():
        if isinstance(child, exclude_module_list):
            continue
        child_result = get_param_names(child, exclude_module_list)
        param_names.extend([f'{child_name}.{n}' for n in child_result])

    param_names.extend(module._parameters.keys())
    return param_names

def clean_line(line: str) -> str:
    line = line.encode('utf-8', 'ignore').decode()  # ignore non-utf-8 characters
    line = line.translate(NON_PRINTABLE_TABLE)
    line = re.sub(r'[\u4e00-\u9fa5]+', '', line)
    line = url_regex.sub('', line)
    line = emoji.replace_emoji(line, '')
    return line.strip()
