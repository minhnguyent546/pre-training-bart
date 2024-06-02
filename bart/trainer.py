from contextlib import nullcontext
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from tokenizers import Tokenizer

from bart.compute_bleu import compute_dataset_bleu
from bart.models import BartBase, BartConfig
import bart.models.utils as model_utils


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
    initial_accum_train_loss: float = 0.0
    beam_size: int = 4
    beam_return_topk: int = 1
    log_sentences: bool = False
    log_sentences_interval: int = 25
    compute_bleu_max_steps: int = 200


class Trainer:
    def __init__(
        self,
        model: BartBase,
        optimizer: torch.optim.Optimizer,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        args: TrainingArguments,
        bart_config: BartConfig,
        lr_scheduler,
        scaler,
        writer: SummaryWriter,
    ) -> None:
        self.model = model
        self.device = model.device
        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.args = args
        self.bart_config = bart_config
        self.lr_scheduler = lr_scheduler
        self.writer = writer

        # mixed precision training with fp16
        self.autocast_ctx = nullcontext()
        self.train_dtype = torch.float32
        if args.fp16 and torch.cuda.is_available() and self.model.device.type == 'cuda':
            self.train_dtype = torch.float16
            self.autocast_ctx = torch.cuda.amp.autocast(dtype=self.train_dtype)

        self.scaler = scaler
        self.accum_train_loss = args.initial_accum_train_loss

    def train(
        self,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
    ) -> None:
        # set model in training mode
        self.model.train()

        global_step = self.args.initial_global_step
        train_progress_bar = tqdm(range(global_step, self.args.train_steps), desc='Training model')
        batch_loss = 0.0
        while global_step < self.args.train_steps:
            torch.cuda.empty_cache()

            for batch_idx, batch in enumerate(train_data_loader):
                input_ids = batch['input_ids'].to(self.device).type(torch.int32)
                input_mask = batch['input_mask'].to(self.device).type(torch.int32)
                labels = batch['labels'].to(self.device).type(torch.int64)
                self.optimizer.zero_grad()

                with self.autocast_ctx:
                    outputs = self.model(
                        encoder_input_ids=input_ids,
                        encoder_attn_mask=input_mask,
                        labels=labels,
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

                    for group_id, group_lr in enumerate(self.lr_scheduler.get_last_lr()):
                        self.writer.add_scalar(f'learning_rate/group-{group_id}', group_lr, global_step)

                    self.lr_scheduler.step()

                    train_progress_bar.set_postfix({'loss': f'{batch_loss:0.3f}'})

                    self.writer.add_scalar('loss/batch_loss', batch_loss, global_step)
                    self.writer.flush()

                    self.accum_train_loss += batch_loss
                    batch_loss = 0.0

                    if (global_step + 1) % self.args.valid_interval == 0:
                        self._valid_step(global_step, valid_data_loader)

                    if (global_step + 1) % self.args.save_interval == 0:
                        self._save_checkpoint(global_step + 1)

                    global_step += 1
                    train_progress_bar.update()
                    if global_step >= self.args.train_steps:
                        break

    def _valid_step(self, global_step: int, valid_data_loader: DataLoader):
        valid_results = model_utils.eval_model(self.model, valid_data_loader, self.device)
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
            compute_bleu_max_steps=self.args.compute_bleu_max_steps,
        )
        self.writer.add_scalars('loss', {
            'train': self.accum_train_loss / self.args.valid_interval,
            'valid': valid_results['loss'],
        }, global_step + 1)
        self.writer.add_scalar('valid_bleu', valid_bleu, global_step + 1)
        self.writer.flush()
        self.accum_train_loss = 0.0

    def _save_checkpoint(
        self,
        global_step: int,
    ) -> None:
        checkpoint_dict = {
            'global_step': global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'config': self.bart_config,
            'accum_train_loss': self.accum_train_loss,
        }
        model_utils.ensure_num_saved_checkpoints(
            self.args.checkpoints_dir,
            self.args.model_basename,
            self.args.saved_checkpoints_limit - 1,
        )
        model_save_path = os.path.join(self.args.checkpoints_dir, f'{self.args.model_basename}-{global_step}.pt')
        torch.save(checkpoint_dict, model_save_path)
