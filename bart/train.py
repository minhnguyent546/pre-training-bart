"""
Pre-training BART with denoising objective.

requires: python >= 3.10
"""

import argparse
from contextlib import nullcontext
import os
from tqdm.autonotebook import tqdm

import datasets
from tokenizers import Tokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from constants import SpecialToken
from models import (
    BartBase,
    BartConfig,
    BartForGeneration,
    LayerNormalization,
)
import opts
import utils


def train_model(args: argparse.Namespace):
    checkpoints_dir = utils.ensure_dir(args.checkpoints_dir)

    # load trained tokenizer
    src_tokenizer: Tokenizer = Tokenizer.from_file(os.path.join(checkpoints_dir, args.tokenizer_basename))
    target_tokenizer = src_tokenizer

    # create data loaders
    saved_dataset = load_saved_dataset(args.data_file_format, args.data_file, args.test_size)
    saved_dataset = saved_dataset.with_format('torch')
    train_data_loader = DataLoader(
        saved_dataset['train'],
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        saved_dataset['test'],
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

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
            num_heads=args.num_heads,
            num_hidden_layers=args.num_hidden_layers,
            intermediate_size=args.intermediate_size,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            activation=args.activation,
            pooler_dropout=args.pooler_dropout,
            pooler_activation=args.pooler_activation,
        )
    else:
        print(f'Loading states from checkpoint {args.from_checkpoint}')

        checkpoint_states = torch.load(args.from_checkpoint, map_location=device)
        required_keys = ['model', 'optimizer', 'lr_scheduler', 'config']
        for key in required_keys:
            if key not in checkpoint_states:
                raise ValueError(f'Missing key "{key}" in checkpoint')
        bart_config = checkpoint_states['config']

    # model, optimizer, lr_scheduler, criterion
    model = BartForGeneration(bart_config)
    model.to(device)
    print(f'Model has {model.num_params()} parameters')
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
        lr_lambda=lambda step: learning_rate * utils.noam_decay(
            step,
            args.hidden_size,
            args.warmup_steps
        ),
    )
    initial_global_step = 0
    accum_train_loss = 0.0
    if checkpoint_states is not None:
        model.load_state_dict(checkpoint_states['model'])
        optimizer.load_state_dict(checkpoint_states['optimizer'])
        lr_scheduler.load_state_dict(checkpoint_states['lr_scheduler'])
        if 'global_step' in checkpoint_states:
            initial_global_step = checkpoint_states['global_step']
        if 'accum_train_loss' in checkpoint_states:
            accum_train_loss = checkpoint_states['accum_train_loss']

    # mixed precision training with fp16
    train_dtype = torch.float32
    autocast_context = nullcontext()
    if args.fp16 and torch.cuda.is_available() and device.type == 'cuda':
        train_dtype = torch.float16
        autocast_context = torch.cuda.amp.autocast(dtype=train_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(train_dtype == torch.float16))

    # tensorboard
    writer = SummaryWriter(args.expr_name)

    # training loop
    train_steps = args.train_steps
    valid_interval = args.valid_interval
    save_interval = args.save_interval
    model.train()

    train_progress_bar = tqdm(
        range(initial_global_step, train_steps),
        desc='Training model',
        position=0,
    )
    global_step = initial_global_step
    while global_step < train_steps:
        torch.cuda.empty_cache()

        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device).type(torch.int32)
            input_mask = batch['input_mask'].to(device).type(torch.int32)
            labels = batch['labels'].to(device).type(torch.int64)
            optimizer.zero_grad()

            with autocast_context:
                logits, loss = model(input_ids, input_mask=input_mask, labels=labels)

            scaler.scale(loss).backward()

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
                writer.add_scalar(f'learning_rate/group-{group_id}', group_lr, global_step)

            lr_scheduler.step()

            train_progress_bar.set_postfix({'loss': f'{loss.item():0.3f}'})
            accum_train_loss += loss.item()

            writer.add_scalar('loss/batch_loss', loss.item(), global_step)
            writer.flush()

            if (global_step + 1) % valid_interval == 0:
                valid_results = eval_model(model, test_data_loader, device)
                writer.add_scalars('loss/', {
                    'train': accum_train_loss / valid_interval,
                    'valid': valid_results['mlm_loss'],
                }, global_step + 1)
                writer.flush()
                accum_train_loss = 0.0

            if (global_step + 1) % save_interval == 0:
                checkpoint_dict = {
                    'global_step': global_step + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': bart_config,
                    'accum_train_loss': accum_train_loss,
                }
                model_save_path = os.path.join(checkpoints_dir, f'bart-{global_step + 1}.pt')
                torch.save(checkpoint_dict, model_save_path)

            global_step += 1
            train_progress_bar.update()
            if global_step >= train_steps:
                break

def eval_model(
    model: BartBase,
    eval_data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    is_training = model.training
    model.eval()

    accum_valid_loss = 0.0
    batch_iter = tqdm(eval_data_loader, desc='Evaluating model')
    with torch.no_grad():
        for batch in batch_iter:
            input_ids = batch['input_ids'].to(device).type(torch.int32)
            input_mask = batch['input_mask'].to(device).type(torch.int32)
            labels = batch['labels'].to(device).type(torch.int64)

            logits, loss = model(input_ids, input_mask, labels)
            accum_valid_loss += loss.item()

            batch_iter.set_postfix({'loss': f'{loss.item():0.3f}'})

    model.train(is_training)

    num_iterations = len(eval_data_loader)
    return {
        'loss': accum_valid_loss / num_iterations,
    }

def load_saved_dataset(data_file_format: str, data_files, test_size: int) -> datasets.DatasetDict:
    raw_dataset: datasets.DatasetDict = datasets.load_dataset(data_file_format, data_files=data_files)
    dataset = raw_dataset['train'].train_test_split(test_size=test_size, shuffle=True)
    return dataset

def main():
    parser = argparse.ArgumentParser(
        description='Pre-training BART model with denoising objective',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.train_opts(parser)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    train_model(args)


if __name__ == '__main__':
    main()
