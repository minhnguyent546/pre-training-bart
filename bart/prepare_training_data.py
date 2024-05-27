"""
Prepare data for pre-training BART with denoising objective.

requires: python >= 3.10
"""

import argparse
from dataclasses import dataclass
import math
import os
import random
from typing import TypeAlias, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tokenizers import Tokenizer

from torch import Tensor
import torch

from tqdm import tqdm

from constants import SpecialToken, WORD_PIECE_SUBWORD_PREFIX
import opts
import utils


TokenType: TypeAlias = str
TokenList: TypeAlias = list[TokenType]

@dataclass
class TrainingInstance:
    source: TokenList
    target: TokenList

def create_training_instances(
    data_files: list[str],
    tokenizer: Tokenizer,
    src_seq_length: int,
    target_seq_length: int,
    num_rounds: int,
    masking_ratio: float,
    deletion_ratio: float,
    infilling_ratio: float,
    permutation_ratio: float,
    rotation_ratio: float,
    span_lengths_lambda: float,
    short_seq_prob: float,
    whole_word_masking: bool,
) -> list[TrainingInstance]:
    documents: list[list[TokenList]] = [[]]
    for data_file in data_files:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'Reading data from {data_file}', unit='lines'):
                line = utils.clean_line(line)
                if not line:
                    documents.append([])
                    continue
                tokens = tokenizer.encode(line)
                if not tokens.tokens:  # a string contains only spaces will be encoded to an empty list!
                    continue
                documents[-1].append(tokens.tokens)

    # remove empty documents
    documents = [doc for doc in documents if doc]
    training_instances = []

    # create mask span distribution using poisson distribution
    mask_span_distribution = create_poisson_distribution(span_lengths_lambda, src_seq_length)

    for round_idx in range(num_rounds):
        docs_iter = tqdm(range(len(documents)), desc=f'Creating training instances {round_idx + 1}-th iteration')
        for doc_idx in docs_iter:
            training_instances.extend(create_training_instances_from_doc(
                documents,
                doc_idx,
                src_seq_length,
                tokenizer,
                masking_ratio,
                deletion_ratio,
                infilling_ratio,
                permutation_ratio,
                rotation_ratio,
                mask_span_distribution,
                short_seq_prob,
                whole_word_masking,
            ))
    return training_instances

def create_training_instances_from_doc(
    documents: list[list[TokenList]],
    doc_idx: int,
    src_seq_length: int,
    tokenizer: Tokenizer,
    masking_ratio: float,
    deletion_ratio: float,
    infilling_ratio: float,
    permutation_ratio: float,
    rotation_ratio: float,
    mask_span_distribution: torch.distributions.Categorical,
    short_seq_prob: float,
    whole_word_masking: bool,
) -> list[TrainingInstance]:
    doc = documents[doc_idx]
    max_num_tokens = src_seq_length - 2  # excluding <s>, </s>

    # 10% of the time, we will use shorter sequences to minimize mismatch between
    # pre-training and fine-tuning
    if np.random.random() < short_seq_prob:
        max_num_tokens = random.randint(2, max_num_tokens)

    training_instances: list[TrainingInstance] = []

    flat_doc: TokenList = []
    for segment in doc:
        flat_doc.extend(segment)
    doc_size = len(flat_doc)
    for idx in range(0, doc_size, max_num_tokens):
        source = [SpecialToken.SOS] + flat_doc[idx:idx+max_num_tokens] + [SpecialToken.EOS]
        target = source[1:-1] + [SpecialToken.EOS]
        if masking_ratio > 0.0:
            source = add_mask_with_span(
                source,
                masking_ratio,
                tokenizer,
                whole_word_masking,
                mask_span_distribution=None,
            )
        if deletion_ratio > 0.0:
            source = add_deletion_noise(source, deletion_ratio, tokenizer)
        if infilling_ratio > 0.0:
            source = add_mask_with_span(
                source,
                infilling_ratio,
                tokenizer,
                whole_word_masking,
                mask_span_distribution=mask_span_distribution,
            )
        if rotation_ratio > 0.0 and np.random.random() < rotation_ratio:
            source = add_rolling_noise(source)
        if permutation_ratio > 0.0:
            source = permute_sentences(source, permutation_ratio)

        training_instances.append(TrainingInstance(source=source, target=target))

    return training_instances

def write_training_instances_to_file(
    output_file: str,
    format: Literal['csv', 'parquet'],
    training_instances: list[TrainingInstance],
    tokenizer: Tokenizer,
    src_seq_length: int,
    target_seq_length: int,
    save_tokens: bool = False,
) -> None:
    headers = ['input_ids', 'input_mask', 'labels']
    if save_tokens:
        headers.extend(['source_tokens', 'target_tokens'])
    content = []
    pad_token_id = tokenizer.token_to_id(SpecialToken.PAD)
    for instance in training_instances:
        source, target = instance.source, instance.target
        input_ids = [tokenizer.token_to_id(token) for token in source]
        labels = [tokenizer.token_to_id(token) for token in target]
        input_mask = [1] * len(input_ids)

        assert len(input_ids) <= src_seq_length
        while len(input_ids) < src_seq_length:
            source.append(SpecialToken.PAD)
            input_ids.append(pad_token_id)
            input_mask.append(0)

        assert len(labels) <= target_seq_length
        while len(labels) < target_seq_length:
            labels.append(pad_token_id)
            target.append(SpecialToken.PAD)

        content.append([input_ids, input_mask, labels])
        if save_tokens:
            source_tokens = instance.source
            target_tokens = instance.target
            content[-1].extend([source_tokens, target_tokens])

    df = pd.DataFrame(content, columns=headers)
    if format == 'csv':
        df.to_csv(output_file, index=False)
    elif format == 'parquet':
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
    else:
        raise ValueError(f'Unsupported format: {format}')

    print(f'Wrote {len(content)} training instances to {output_file}')

def add_mask_with_span(
    tokens: TokenList,
    mask_ratio: float,
    tokenizer: Tokenizer,
    whole_word_masking: bool,
    mask_span_distribution: torch.distributions.Categorical | None = None,
) -> TokenList:
    """
    If `mask_span_distribution` is not provided, it is equal to token masking,
    otherwise it is text infilling.
    """
    cand_indices: list[list[int]] = []
    for idx, token in enumerate(tokens):
        if token in (SpecialToken.SOS, SpecialToken.EOS):
            continue
        if whole_word_masking and len(cand_indices) > 0 and token.startswith(WORD_PIECE_SUBWORD_PREFIX):
            cand_indices[-1].append(idx)
        else:
            cand_indices.append([idx])

    num_to_infill = int(round(len(cand_indices) * mask_ratio))

    if mask_span_distribution is not None:
        lengths = np.array([])
        while lengths.size < num_to_infill:
            aux_lengths = mask_span_distribution.sample((num_to_infill,)).numpy()
            lengths = np.concatenate((lengths, aux_lengths)).astype(int)

        # cutting down to `num_to_infill`
        i = 0
        cum_length = 0
        while i < lengths.size and cum_length <= num_to_infill:
            assert i < lengths.size
            cum_length += lengths[i]
            i += 1
        if cum_length > num_to_infill:
            cum_length -= lengths[i - 1]
            i -= 1
        assert cum_length <= num_to_infill
        lengths = lengths[:i]
        num_to_infill = cum_length
        lengths = np.array(lengths)
    else:
        lengths = np.ones(num_to_infill, dtype=int)

    # handle insertion (i.e. length = 0) separately
    num_to_insert = lengths[lengths == 0].size
    lengths = lengths[lengths > 0]

    # now, let's mask the contiguous spans
    result_tokens: TokenList = tokens[:]
    lptr, rptr = 0, -1
    cand_sz = len(cand_indices)
    consumed_length = 0

    for length in lengths:
        assert lptr < cand_sz
        consumed_length += length

        # expand rptr to the right as far as we can
        while rptr + 1 < cand_sz and cand_sz - rptr - 2 >= num_to_infill - consumed_length:
            rptr += 1

        # weighted random
        start_cand_idx = utils.wnext(lptr, rptr - length, time=-np.random.randint(1, 2))
        # start_cand_idx = utils.wnext(lptr, rptr - length, time=-np.random.randint(10, 16))

        # all tokens from [start_cand_idx, start_cand_idx + length - 1] will be masked
        for j in range(length):
            for idx in cand_indices[start_cand_idx + j]:
                result_tokens[idx] = SpecialToken.PLACEHOLDER
        # we will use Bert's mask strategy 80/0/20 here
        # (corresponding to replace with mask/keep/random)
        new_token = None
        if np.random.random() < 0.8:
            new_token = SpecialToken.MASK
        else:
            new_token = tokenizer.id_to_token(np.random.randint(0, tokenizer.get_vocab_size()))

        result_tokens[cand_indices[start_cand_idx][0]] = new_token
        lptr = start_cand_idx + length

    result_tokens = [token for token in result_tokens if token != SpecialToken.PLACEHOLDER]
    if num_to_insert > 0:
        result_tokens = add_insertion_noise(result_tokens, num_to_insert / len(tokens), tokenizer)

    return result_tokens

def add_deletion_noise(tokens: TokenList, deletion_ratio: float, tokenizer: Tokenizer) -> TokenList:
    num_deletions = int((len(tokens) - 2) * deletion_ratio)
    indices_to_delete = np.random.permutation(len(tokens) - 2)[:num_deletions]
    indices_to_delete += 1  # add an offset to skip SOS token
    delete_mask = np.zeros(len(tokens), dtype=bool)
    delete_mask[indices_to_delete] = True
    result_tokens = np.array(tokens)[~delete_mask].tolist()
    assert len(result_tokens) == len(tokens) - num_deletions
    return result_tokens

def add_insertion_noise(tokens: TokenList, insertion_ratio: float, tokenizer: Tokenizer) -> TokenList:
    sz = len(tokens)
    num_to_insert = int((sz - 2) * insertion_ratio)
    insert_indices = np.random.permutation(sz - 2 + num_to_insert)[:num_to_insert] + 1  # add an offset to skip SOS token
    insert_mask = np.zeros(sz + num_to_insert, dtype=bool)
    insert_mask[insert_indices] = True
    result_tokens = np.full((sz + num_to_insert,), SpecialToken.PLACEHOLDER, dtype=object)

    # we will use Bert's mask strategy 80/0/20 here
    # (corresponding to replace with mask/keep/random)
    num_masked_tokens = int(num_to_insert * 0.8)
    result_tokens[insert_indices[:num_masked_tokens]] = SpecialToken.MASK
    result_tokens[insert_indices[num_masked_tokens:]] = [
        tokenizer.id_to_token(token_id)
        for token_id in np.random.randint(
            0,
            tokenizer.get_vocab_size(),
            size=(num_to_insert - num_masked_tokens,),
        )
    ]

    result_tokens[~insert_mask] = tokens
    assert (result_tokens != SpecialToken.PLACEHOLDER).all()
    return result_tokens.tolist()

def add_rolling_noise(tokens: TokenList) -> TokenList:
    raise NotImplementedError()

def permute_sentences(tokens: TokenList, permutation_ratio: float) -> TokenList:
    raise NotImplementedError()

def create_poisson_distribution(
    _lambda: float,
    max_num_events: int,
    eps_to_stop: float = 1e-4,
) -> torch.distributions.Categorical:
    e_power_neg_lambda = math.exp(-_lambda)
    lambda_power_k = 1.0
    k_factorial = 1.0
    probs = []
    for k in range(max_num_events):
        event_probs = e_power_neg_lambda * lambda_power_k / k_factorial
        probs.append(event_probs)
        lambda_power_k *= _lambda
        k_factorial *= (k + 1)
        if event_probs < eps_to_stop:
            break

    distribution = torch.distributions.Categorical(Tensor(probs).float())
    return distribution

def main():
    parser = argparse.ArgumentParser(
        description='Building tokenizer and preparing training instances',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opts.prepare_training_data_opts(parser)
    args = parser.parse_args()

    if args.target_seq_length < args.src_seq_length:
        raise ValueError(
            'Target sequence length cannot be smaller than source sequence length, '
            f'but got {args.target_seq_length} and {args.src_seq_length}.'
        )

    utils.set_random_seed(args.seed)

    checkpoints_dir = utils.ensure_dir(args.checkpoints_dir)
    tokenizer_save_path = os.path.join(checkpoints_dir, args.tokenizer_basename)
    tokenizer = Tokenizer.from_file(tokenizer_save_path)
    training_instances = create_training_instances(
        args.data_file,
        tokenizer,
        args.src_seq_length,
        args.target_seq_length,
        args.num_rounds,
        args.masking_ratio,
        args.deletion_ratio,
        args.infilling_ratio,
        args.permutation_ratio,
        args.rotation_ratio,
        args.span_lengths_lambda,
        args.short_seq_prob,
        args.whole_word_masking,
    )
    write_training_instances_to_file(
        args.output_file,
        args.format,
        training_instances,
        tokenizer,
        args.src_seq_length,
        args.target_seq_length,
        args.save_tokens,
    )


if __name__ == '__main__':
    main()
