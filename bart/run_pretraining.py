"""
Pre-training BART with denoising objective.
Requires: python >= 3.10
"""

import argparse

from tokenizers import Tokenizer

import wandb

import torch
from torch.utils.data import DataLoader

from bart import opts, utils
from bart.constants import SpecialToken
from bart.models import (
    BartConfig,
    BartForGeneration,
    LayerNormalization,
)
from bart.trainer import Trainer, TrainingArguments


def train_model(args: argparse.Namespace):

    # loading pre-trained tokenizers
    src_tokenizer: Tokenizer = Tokenizer.from_file(args.src_tokenizer)
    target_tokenizer: Tokenizer = Tokenizer.from_file(args.target_tokenizer)

    # loading datasets
    raw_dataset = utils.load_dataset_from_files(
        args.data_file_format,
        data_files={'train': args.train_files, 'test': args.test_files, 'validation': args.validation_files},
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.split_dataset_seed,
        field=args.field,
    )
    # TODO: check if raw_dataset contains 'train' and 'test' split
    assert 'train' in raw_dataset and 'test' in raw_dataset

    # creating data loaders
    raw_dataset = raw_dataset.with_format('torch')
    train_data_loader = DataLoader(
        raw_dataset['train'],
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        raw_dataset['test'],
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    use_fp16 = args.fp16 and device.type == 'cuda'

    checkpoint_states = None
    if args.from_checkpoint is None:
        print('Starting training from scratch')
        bart_config = BartConfig(
            src_pad_token_id=src_tokenizer.token_to_id(SpecialToken.PAD),
            target_pad_token_id=target_tokenizer.token_to_id(SpecialToken.PAD),
            target_start_token_id=target_tokenizer.token_to_id(SpecialToken.SOS),
            target_end_token_id=target_tokenizer.token_to_id(SpecialToken.EOS),
            src_vocab_size=src_tokenizer.get_vocab_size(),
            target_vocab_size=target_tokenizer.get_vocab_size(),
            src_seq_length=args.src_seq_length,
            target_seq_length=args.target_seq_length,
            max_position_embeddings=args.max_position_embeddings,
            device=device,
            shared_vocab=args.shared_vocab,
            tie_weights=args.tie_weights,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            encoder_num_heads=args.encoder_num_heads,
            encoder_num_hidden_layers=args.encoder_num_hidden_layers,
            decoder_num_heads=args.decoder_num_heads,
            decoder_num_hidden_layers=args.decoder_num_hidden_layers,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            activation=args.activation,
            pre_norm=args.pre_norm,
            pooler_dropout=args.pooler_dropout,
            pooler_activation=args.pooler_activation,
        )
    else:
        print(f'Loading states from checkpoint {args.from_checkpoint}')

        checkpoint_states = torch.load(args.from_checkpoint, map_location=device)
        required_keys = ['model', 'optimizer', 'lr_scheduler', 'config']
        if use_fp16:
            required_keys.append('scaler')
        for key in required_keys:
            if key not in checkpoint_states:
                raise ValueError(f'Missing key "{key}" in checkpoint')
        bart_config = checkpoint_states['config']
        bart_config.device = device

    # model, optimizer, lr_scheduler, scaler
    model = BartForGeneration(bart_config)
    model.to(device)
    learning_rate = args.learning_rate
    optimizer = utils.make_optimizer(
        model,
        args.optim,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        exclude_module_list=(LayerNormalization,),
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate * args.accum_step * utils.noam_decay(
            step,
            args.hidden_size,
            args.warmup_steps
        ),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    initial_global_step = 0
    initial_accum_train_loss = 0.0
    if checkpoint_states is not None:
        model.load_state_dict(checkpoint_states['model'])
        optimizer.load_state_dict(checkpoint_states['optimizer'])
        lr_scheduler.load_state_dict(checkpoint_states['lr_scheduler'])
        scaler.load_state_dict(checkpoint_states['scaler'])
        if 'global_step' in checkpoint_states:
            initial_global_step = checkpoint_states['global_step']
        if 'accum_train_loss' in checkpoint_states:
            initial_accum_train_loss = checkpoint_states['accum_train_loss']

    # training arguments
    training_args = TrainingArguments(
        checkpoints_dir=args.checkpoints_dir,
        saved_checkpoints_limit=args.saved_checkpoints_limit,
        train_steps=args.train_steps,
        valid_interval=args.valid_interval,
        save_interval=args.save_interval,
        accum_step=args.accum_step,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        initial_global_step=initial_global_step,
        initial_accum_train_loss=initial_accum_train_loss,
        beam_size=args.beam_size,
        beam_return_topk=args.beam_return_topk,
        log_sentences=args.log_sentences,
        log_sentences_interval=args.log_sentences_interval,
        compute_bleu_max_steps=args.compute_bleu_max_steps,
    )

    # wandb
    wb_run = None
    if not args.disable_wandb:
        all_config = vars(bart_config) | vars(training_args)
        wb_run = wandb.init(
            project=args.project_name,
            name=args.expr_name,
            config=all_config,
            id=args.wandb_resume_id,
            resume='must' if args.wandb_resume_id is not None else None,
        )

    # trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        src_tokenizer=src_tokenizer,
        target_tokenizer=target_tokenizer,
        args=training_args,
        bart_config=bart_config,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        wb_run=wb_run,
    )
    print(f'Model has {model.num_params()} parameters')
    trainer.train(train_data_loader, test_data_loader)

def main():
    parser = argparse.ArgumentParser(
        description='Pre-training BART model with denoising objective',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.pretrain_opts(parser)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    train_model(args)


if __name__ == '__main__':
    main()
