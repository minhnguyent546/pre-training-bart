from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

from wandb.sdk.wandb_run import Run as WbRun

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from tokenizers import Tokenizer

import bart.models.utils as model_utils
from bart.compute_bleu import compute_dataset_bleu
from bart.models import BartBase, BartConfig


@dataclass
class TrainingArguments():
    checkpoints_dir: str
    model_basename: str = 'bart'
    saved_checkpoints_limit: int = 6
    train_steps: int = 40_000
    valid_interval: int = 3_000
    save_interval: int = 4_000
    accum_step: int = 1
    train_batch_size: int = 32
    eval_batch_size: int = 32
    fp16: bool = False
    label_smoothing: float = 0.0
    max_grad_norm: float = 0.0
    initial_global_step: int = 0
    initial_running_loss: AverageMeter | None = None
    beam_size: int = 4
    beam_return_topk: int = 1
    log_sentences: bool = False
    log_sentences_interval: int = 25
    compute_bleu_max_steps: int = 200
    ddp: bool = False
    is_master: bool = True
    rank: int = -1
    local_rank: int = -1
    master_rank: int = -1
    world_size: int = 0

class Trainer:
    def __init__(
        self,
        model: BartBase | DDP,
        optimizer: torch.optim.Optimizer,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        args: TrainingArguments,
        bart_config: BartConfig,
        lr_scheduler,
        scaler,
        wb_run: WbRun | None = None,
    ) -> None:
        self.model = model
        self.device = model.device
        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.args = args
        self.bart_config = bart_config
        self.lr_scheduler = lr_scheduler
        self.wb_run = wb_run

        # mixed precision training with fp16
        self.autocast_ctx = nullcontext()
        self.train_dtype = torch.float32
        if args.fp16 and torch.cuda.is_available() and self.model.device.type == 'cuda':
            self.train_dtype = torch.float16
            self.autocast_ctx = torch.cuda.amp.autocast(dtype=self.train_dtype)

        self.scaler = scaler
        self.running_loss = args.initial_running_loss
        if self.running_loss is None:
            self.running_loss = AverageMeter('running_loss', device=self.device)

    def train(
        self,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        train_sampler=None,
    ) -> None:
        # set model in training mode
        self.model.train()

        global_step = self.args.initial_global_step
        train_progress_bar = tqdm(range(global_step, self.args.train_steps), desc='Training model')
        batch_loss = 0.0
        while global_step < self.args.train_steps:
            torch.cuda.empty_cache()
            if train_sampler is not None:
                train_sampler.set_epoch(global_step // len(train_data_loader))

            for batch_idx, batch in enumerate(train_data_loader):
                input_ids = batch['input_ids'].to(self.device).type(torch.int32)
                labels = batch['labels'].to(self.device).type(torch.int64)
                input_mask = None
                decoder_input_ids = None
                decoder_input_mask = None
                if 'input_mask' in batch:
                    input_mask = batch['input_mask'].to(self.device).type(torch.int32)
                if 'decoder_input_ids' in batch:
                    decoder_input_ids = batch['decoder_input_ids'].to(self.device).type(torch.int32)
                if 'decoder_input_mask' in batch:
                    decoder_input_mask = batch['decoder_input_mask'].to(self.device).type(torch.int32)

                self.optimizer.zero_grad()

                with self.autocast_ctx:
                    outputs = self.model(
                        encoder_input_ids=input_ids,
                        encoder_attn_mask=input_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attn_mask=decoder_input_mask,
                        labels=labels,
                        label_smoothing=self.args.label_smoothing,
                    )
                    loss = outputs.lm_loss
                    if self.args.accum_step > 1:
                        loss = loss / self.args.accum_step
                    batch_loss += loss.item()

                # useful links about gradient accumulation:
                # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
                # https://github.com/huggingface/transformers/blob/121c24efa4453e4e726b5f0b2cf7095b14b7e74e/src/transformers/trainer.py#L801

                # accumulates scaled gradients
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.args.accum_step == 0 or batch_idx + 1 == len(train_data_loader):
                    if self.args.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self._maybe_report_step(batch_loss, step=global_step)
                    self.lr_scheduler.step()

                    train_progress_bar.set_postfix({'loss': f'{batch_loss:0.3f}'})
                    self.running_loss.update(batch_loss)
                    batch_loss = 0.0

                    if (global_step + 1) % self.args.valid_interval == 0:
                        self._valid_step(global_step + 1, valid_data_loader)

                    if (global_step + 1) % self.args.save_interval == 0:
                        self._save_checkpoint(global_step + 1)

                    global_step += 1
                    train_progress_bar.update()
                    if global_step >= self.args.train_steps:
                        break

    def _valid_step(self, step: int, valid_data_loader: DataLoader):
        if self.args.ddp:
            self.running_loss.reduce(dst=self.args.master_rank)
        valid_results = model_utils.eval_model(self.model, valid_data_loader, self.device)
        # TODO: make compute_data_bleu run on multiple GPUs
        if self.args.is_master:
            valid_bleu = compute_dataset_bleu(
                self.model,
                valid_data_loader.dataset,
                self.src_tokenizer,
                self.target_tokenizer,
                self.bart_config.target_seq_length,
                beam_size=self.args.beam_size,
                beam_return_topk=self.args.beam_return_topk,
                log_sentences=self.args.log_sentences,
                logging_interval=self.args.log_sentences_interval,
                max_steps=self.args.compute_bleu_max_steps,
            )
            self._maybe_report_valid_step(valid_results, valid_bleu=valid_bleu, step=step)
        self.running_loss.reset()

    def _save_checkpoint(
        self,
        global_step: int,
    ) -> None:
        if self.args.ddp:
            running_losses = [None for _ in range(self.args.world_size)] if self.args.is_master else None
            dist.gather_object(vars(self.running_loss), running_losses, dst=self.args.master_rank)
        else:
            running_losses = self.running_loss
        if self.args.is_master:
            checkpoint_dict = {
                'global_step': global_step,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'config': self.bart_config,
                'training_args': self.args,
                'running_loss': running_losses,
            }
            model_utils.ensure_num_saved_checkpoints(
                self.args.checkpoints_dir,
                self.args.model_basename,
                self.args.saved_checkpoints_limit - 1,
            )
            model_save_path = os.path.join(self.args.checkpoints_dir, f'{self.args.model_basename}-{global_step}.pt')
            torch.save(checkpoint_dict, model_save_path)

    def _maybe_report_step(self, batch_loss: float, step: int) -> None:
        if self.wb_run is None:
            return

        for group_id, group_lr in enumerate(self.lr_scheduler.get_last_lr()):
            self.wb_run.log({f'learning_rate/group-{group_id}': group_lr}, step=step)

        self.wb_run.log({'loss/batch_loss': batch_loss}, step=step)

    def _maybe_report_valid_step(
        self,
        valid_results: dict[str, Any],
        step: int,
        valid_bleu: float | None = None,
    ) -> None:
        if self.wb_run is None:
            return

        self.wb_run.log({
            'loss/train': self.running_loss.average,
            'loss/valid': valid_results['loss'],
        }, step=step)
        if valid_bleu is not None:
            self.wb_run.log({'valid_bleu': valid_bleu}, step=step)

class AverageMeter:
    """A class for working with average meters."""
    def __init__(
        self,
        name: str,
        value: int | float = 0.0,
        count: int = 0,
        sum: int | float = 0.0,
        device: torch.device | Literal['auto'] = 'auto',
    ) -> None:
        if count == 0:
            value = 0
            sum = 0
        self.name = name
        self.value = value
        self.count = count
        self.sum = sum
        if device == 'auto':
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

    def update(self, value: int | float, nums: int = 1) -> None:
        self.value = value
        self.sum += value * nums
        self.count += nums

    def reduce(self, dst: int) -> None:
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        # only `Tensor` of process with rank `dst` will be modified in-place,
        # `Tensor` of other processes will remain the same
        dist.reduce(meters_to_reduce, dst=dst, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def all_reduce(self) -> None:
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        dist.all_reduce(meters_to_reduce, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    @property
    def average(self) -> float:
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            return 0.0

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def __repr__(self) -> str:
        return (
            f'{self.name}(value={self.value}, '
            f'average={self.average}, '
            f'sum={self.sum}, '
            f'count={self.count}, '
            f'device={self.device})'
        )
