import math
import os
import random
import re
import regex
from typing import Callable, Literal
import unicodedata
import yaml

import numpy as np

import datasets
from tokenizers import Tokenizer

from torch import nn
import torch
from torch.utils.data import DataLoader

from bart.bilingual_dataset import BilingualDataset
from bart.models import BartBase


url_regex = re.compile(r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))")

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

def load_dataset_from_files(
    data_file_format: Literal['json', 'csv', 'parquet', 'arrow'],
    data_files: dict[str, str | list[str] | None],
    test_size: int | None = None,
    validation_size: int | None = None,
    seed: int = 1061109567,
    **kwargs,
) -> datasets.DatasetDict:
    """
    Load dataset from files and split into train, test, and validation sets.

    If testing/validation split does not exist, it will be created from the training set,
    with size `test_size`/`validation_size` if specified.
    """
    data_files = {name: split for name, split in data_files.items() if split}
    if 'train' not in data_files:
        raise ValueError('Training set is required, but not found in `data_files`')
    if data_file_format != 'json':
        kwargs.pop('field', None)
    raw_dataset: datasets.DatasetDict = datasets.load_dataset(
        data_file_format,
        data_files=data_files,
        **kwargs,
    )
    if 'test' not in raw_dataset:
        if test_size is not None:
            if test_size > len(raw_dataset['train']):
                raise ValueError(f'Test size {test_size} is larger than the train set {len(raw_dataset["train"])}')
            raw_dataset = raw_dataset['train'].train_test_split(
                test_size=test_size,
                shuffle=True,
                seed=seed,
            )
    if 'validation' not in raw_dataset:
        if validation_size is not None:
            if validation_size > len(raw_dataset['train']):
                raise ValueError(f'Validation size {validation_size} is larger than the train set {len(raw_dataset["train"])}')
            old_dataset = raw_dataset
            raw_dataset = old_dataset['train'].train_test_split(
                test_size=validation_size,
                shuffle=True,
                seed=seed,
            )
            raw_dataset['validation'] = raw_dataset.pop('test')
            if 'test' in old_dataset:
                raw_dataset['test'] = old_dataset['test']
    return raw_dataset

def noam_decay(step_num: int, d_model: int = 768, warmup_steps: int = 4000):
    """
    As described in https://arxiv.org/pdf/1706.03762.pdf
    """
    step_num = max(step_num, 1)
    return d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

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

def make_bilingual_data_loader(
    dataset: datasets.Dataset,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    src_seq_length: int,
    target_seq_length: int,
    batch_size: int,
    *,
    source_key: str = 'source',
    target_key: str = 'target',
    add_padding_tokens: bool = False,
    include_source_target_text: bool = False,
    shuffle: bool = False,
    pin_memory: bool = False,
    collate_fn: Callable | None = None,
) -> DataLoader:
    bilingual_dataset = BilingualDataset(
        dataset,
        src_tokenizer,
        target_tokenizer,
        src_seq_length,
        target_seq_length,
        source_key=source_key,
        target_key=target_key,
        add_padding_tokens=add_padding_tokens,
        include_source_target_text=include_source_target_text,
    )
    bilingual_data_loader = DataLoader(
        bilingual_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return bilingual_data_loader

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

def clean_text(text: str, *, strip: bool = True, keep_punct: bool = True) -> str:
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    # remove urls
    text = url_regex.sub('', text)
    # remove non-latin characters (but keep numbers, punctuations, and whitespaces)
    if keep_punct:
        text = regex.sub(r'([^\p{Latin}\p{Punctuation}0-9\s]+)', r'', text)
    else:
        text = regex.sub(r'([^\p{Latin}0-9\s]+)', r'', text)
    if strip:
        text = text.strip()
    return text

def get_perplexity(loss: float) -> float:
    return math.exp(loss)
