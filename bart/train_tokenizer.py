"""
Train tokenizer from text files
Requires: python >= 3.10
"""

import argparse
from tqdm.auto import tqdm

from tokenizers import AddedToken, Tokenizer
import tokenizers
import tokenizers.decoders
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.trainers
import tokenizers.normalizers

from bart import opts, utils
from bart.constants import SpecialToken


def train_tokenizer(
    data_iter,
    vocab_size: int = 32_000,
    min_freq: int = 2,
    lowercase: bool = False,
    show_progress: bool = True,
) -> Tokenizer:
    tokenizer = Tokenizer(tokenizers.models.WordPiece(
        unk_token=SpecialToken.UNK,
        max_input_chars_per_word=100,
    ))  # pyright: ignore[reportCallIssue]
    # pre-tokenizer
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    # normalizers
    normalizer_list = []
    if lowercase:
        normalizer_list.append(tokenizers.normalizers.Lowercase())
    if normalizer_list:
        tokenizer.normalizer = tokenizers.normalizers.Sequence(normalizer_list)

    # decoder
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

def build_tokenizer(
    data_files: str | list[str],
    vocab_size: int,
    min_freq: int = 1,
    lowercase: bool = False,
    save_path: str | None = None,
) -> Tokenizer:
    if isinstance(data_files, str):
        data_files = [data_files]
    data = []
    for data_file in data_files:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Reading file', unit=' lines'):
                line = utils.clean_text(line, strip=True, keep_punct=True)
                data.append(line)

    tokenizer = train_tokenizer(
        utils.chunks(data, chunk_size=10_000),
        vocab_size=vocab_size,
        min_freq=min_freq,
        lowercase=lowercase,
    )
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

    build_tokenizer(
        args.data_files,
        args.vocab_size,
        min_freq=args.min_freq,
        lowercase=args.lowercase,
        save_path=args.output,
    )


if __name__ == '__main__':
    main()
