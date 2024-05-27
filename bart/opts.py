import argparse


def train_tokenizer_opts(parser: argparse.ArgumentParser) -> None:
    """All options used in train_tokenizer"""
    _add_general_opts(parser)
    _add_train_tokenizer_opts(parser)

def prepare_training_data_opts(parser: argparse.ArgumentParser) -> None:
    """All options used in prepare_training_data"""
    _add_general_opts(parser)
    _add_data_prepare_opts(parser)

def train_opts(parser):
    """All options used in train"""
    _add_general_opts(parser)
    _add_dataset_opts(parser)
    _add_model_opts(parser)
    _add_training_opts(parser)
    _add_compute_valid_bleu_opts(parser)

def _add_general_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('General options')
    group.add_argument(
        '--checkpoints-dir',
        help='Directory to save checkpoints',
        type=str,
        default='./checkpoints',
    )
    group.add_argument(
        '--tokenizer-basename',
        help='Tokenizer basename',
        type=str,
        default='tokenizer.json',
    )
    group.add_argument(
        '--expr-name',
        help='Experiment name',
        type=str,
        default='runs/bart',
    )
    group.add_argument(
        '--seed',
        help='Random seed',
        type=int,
        default=1061109567,
    )

def _add_dataset_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Dataset')
    group.add_argument(
        '--data-file',
        nargs='+',
        help='Path to the prepared file contains training instances',
        type=str,
    )
    group.add_argument(
        '--data-file-format',
        help='Data file format',
        type=str,
        default='csv',
    )
    group.add_argument(
        '--test-size',
        help='Test size',
        type=int,
        default=10_000,
    )

def _add_model_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--src-seq-length',
        help='Maximum source sequence length',
        type=int,
        default=128,
    )
    group.add_argument(
        '--target-seq-length',
        help='Maximum target sequence length',
        type=int,
        default=156,
    )
    group.add_argument(
        '--max-position-embeddings',
        help='Maximum sequence length for position embeddings',
        type=int,
        default=512,
    )
    group.add_argument(
        '--shared-vocab',
        help='Whether to share vocabulary between source and target',
        action='store_true',
    )
    group.add_argument(
        '--tie-weights',
        help='Whether to use tied weights for embeddings and output layer',
        action='store_true',
    )
    group.add_argument(
        '--hidden-size',
        help='Hidden size (i.e. size of embedding vectors)',
        type=int,
        default=768,
    )
    group.add_argument(
        '--num-hidden-layers',
        help='Number of hidden layers in the encoder block',
        type=int,
        default=6,
    )
    group.add_argument(
        '--num-heads',
        help='Number of attention heads',
        type=int,
        default=12,
    )
    group.add_argument(
        '--intermediate-size',
        help='Intermediate size of the feed-forward network',
        type=int,
        default=768 * 4,
    )
    group.add_argument(
        '--dropout',
        help='Dropout rate',
        type=float,
        default=0.1,
    )
    group.add_argument(
        '--attn-dropout',
        help='Dropout rate in attention',
        type=float,
        default=0.1,
    )
    group.add_argument(
        '--activation',
        help='Activation type',
        type=str,
        choices=['relu', 'gelu'],
        default='gelu',
    )
    group.add_argument(
        '--pooler-dropout',
        help='Dropout rate in pooler',
        type=float,
        default=0.1,
    )
    group.add_argument(
        '--pooler-activation',
        help='Activation used in pooler',
        type=float,
        choices=['tanh'],
        default=0.1,
    )

def _add_data_prepare_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Prepare data')
    group.add_argument(
        '--data-file',
        nargs='+',
        help='Path to the text files contain documents',
        required=True,
        type=str,
    )
    group.add_argument(
        '--output-file',
        help='Path where to save training instances',
        type=str,
        default='./datasets/training_instances.csv',
    )
    group.add_argument(
        '--format',
        help='Format of the saved output file',
        type=str,
        choices=['csv', 'parquet'],
        default='csv',
    )
    group.add_argument(
        '--save-tokens',
        help='Whether to save tokens along with token ids',
        action='store_true',
    )
    group.add_argument(
        '--num-rounds',
        help='Number of times to duplicate the input data (with different masks)',
        type=int,
        default=5,
    )
    group.add_argument(
        '--src-seq-length',
        help='Maximum source sequence length',
        type=int,
        default=128,
    )
    group.add_argument(
        '--target-seq-length',
        help='Maximum target sequence length',
        type=int,
        default=156,
    )
    group.add_argument(
        '--masking-ratio',
        help='Masking ratio',
        type=float,
        default=0.15,
    )
    group.add_argument(
        '--deletion-ratio',
        help='Deletion ratio',
        type=float,
        default=0.0,
    )
    group.add_argument(
        '--infilling-ratio',
        help='Infilling ratio',
        type=float,
        default=0.0,
    )
    group.add_argument(
        '--permutation-ratio',
        help='Permutation ratio',
        type=float,
        default=0.0,
    )
    group.add_argument(
        '--rotation-ratio',
        help='Rotation ratio',
        type=float,
        default=0.0,
    )
    group.add_argument(
        '--span-lengths-lambda',
        help='Mean of the spanned lengths',
        type=int,
        default=3,
    )
    group.add_argument(
        '--short-seq-prob',
        help='Probability of creating sequences shorter than the maximum length',
        type=float,
        default=0.1,
    )
    group.add_argument(
        '--whole-word-masking',
        help='Whether to mask the whole words instead of subwords',
        action='store_true',
    )

def _add_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Training')
    group.add_argument(
        '--from-checkpoint',
        help='Start training from this checkpoint',
        type=str,
    )
    group.add_argument(
        '--optim',
        help='Optmizer',
        type=str,
        choices=['adam', 'adamw'],
        default='adamw',
    )
    group.add_argument(
        '--weight-decay',
        help='Weight decay',
        type=float,
        default='0.0',
    )
    group.add_argument(
        '--learning-rate',
        help='Initial learning rate',
        type=float,
        default=0.5,
    )
    group.add_argument(
        '--warmup-steps',
        help='Warmup steps',
        type=int,
        default=4_000,
    )
    group.add_argument(
        '--train-batch-size',
        help='Train batch size',
        type=int,
        default=32,
    )
    group.add_argument(
        '--eval-batch-size',
        help='Evaluation batch size',
        type=int,
        default=32,
    )
    group.add_argument(
        '--fp16',
        help='Whether to use mixed precision training with fp16',
        action='store_true',
    )
    group.add_argument(
        '--train-steps',
        help='Number of training steps',
        type=int,
        default=100_000,
    )
    group.add_argument(
        '--valid-interval',
        help='Validation interval',
        type=int,
        default=3_000,
    )
    group.add_argument(
        '--save-interval',
        help='Interval between saving checkpoints',
        type=int,
        default=4_000,
    )
    group.add_argument(
        '--max-grad-norm',
        help='Maximum gradient norm for gradient clipping',
        type=float,
        default=1.0,
    )

def _add_compute_valid_bleu_opts(parser):
    group = parser.add_argument_group('Computing validation BLEU')
    group.add_argument(
        '--beam-size',
        help='Beam size for beam search',
        type=int,
        default=1,
    )
    group.add_argument(
        '--beam-return-topk',
        help='Number of top-k results to return',
        type=int,
        default=1,
    )
    group.add_argument(
        '--log-sentences',
        help='Whether to log sentences during evaluation',
        action='store_true',
    )
    group.add_argument(
        '--logging-interval',
        help='Logging interval',
        type=int,
        default=500,
    )
    group.add_argument(
        '--compute-bleu-max-steps',
        help='Maximum steps to compute BLEU',
        type=int,
    )

def _add_train_tokenizer_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Train tokenizer')
    group.add_argument(
        '--data-file',
        nargs='+',
        help='Path to the text files contain documents',
        type=str,
    )
    group.add_argument(
        '--vocab-size',
        help='Vocabulary size limit',
        type=int,
        default=32_000,
    )
