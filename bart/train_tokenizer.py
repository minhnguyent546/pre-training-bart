"""
Train tokenizer from text files
requires: python >= 3.10
"""

import argparse
import os
from tqdm.auto import tqdm

from tokenizers import AddedToken, Tokenizer
import tokenizers
import tokenizers.decoders
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.trainers

from bart import opts, utils
from bart.constants import SpecialToken


def train_tokenizer(
    data_iter,
    vocab_size: int = 30_000,
    min_freq: int = 1,
    show_progress: bool = True
) -> Tokenizer:
    tokenizer = Tokenizer(tokenizers.models.WordPiece(
        unk_token=SpecialToken.UNK,
        max_input_chars_per_word=100,
    ))  # pyright: ignore[reportCallIssue]
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.decoder = tokenizers.decoders.WordPiece(
        prefix='##',
        cleanup=False,
    )
    trainer = tokenizers.trainers.WordPieceTrainer(
        vocab_size=vocab_size - 1,
        min_frequency=min_freq,
        show_progress=show_progress,
        special_tokens=SpecialToken.all(),
        continuing_subword_prefix='##'
    )
    tokenizer.train_from_iterator(data_iter, trainer=trainer)
    tokenizer.add_special_tokens([AddedToken("\n")])
    return tokenizer

def build_tokenizer(data_files: str | list[str], vocab_size: int, save_path: str | None = None) -> Tokenizer:
    if isinstance(data_files, str):
        data_files = [data_files]
    data = []
    for data_file in data_files:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Reading file', unit='lines'):
                line = utils.clean_line(line)
                data.append(line)

    tokenizer = train_tokenizer(utils.chunks(data, chunk_size=10_000), vocab_size=vocab_size)
    print(f'Vocab size: {tokenizer.get_vocab_size()}')

    if save_path is not None:
        tokenizer.save(save_path)
        print(f'Tokenizer saved to {save_path}')
    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description='Training tokenizer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opts.train_tokenizer_opts(parser)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)

    checkpoints_dir = utils.ensure_dir(args.checkpoints_dir)
    tokenizer_save_path = os.path.join(checkpoints_dir, args.tokenizer_basename)
    build_tokenizer(args.data_file, args.vocab_size, tokenizer_save_path)


if __name__ == '__main__':
    main()
