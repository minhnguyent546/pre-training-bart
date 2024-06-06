"""
Fine-tuning BART for neural machine translation task.
Requires: python >= 3.10
"""

import argparse

from tokenizers import Tokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bart import opts, utils
from bart.bilingual_dataset import BilingualDataset, CollatorWithPadding
from bart.constants import SpecialToken
from bart.models import BartForNMT, BartForNMTConfig, LayerNormalization
from bart.trainer import Trainer, TrainingArguments


def train_model(args: argparse.Namespace):

    # loading pre-trained tokenizers
    src_tokenizer: Tokenizer = Tokenizer.from_file(args.src_tokenizer)
    target_tokenizer: Tokenizer = Tokenizer.from_file(args.target_tokenizer)
    assert src_tokenizer.token_to_id(SpecialToken.PAD) == target_tokenizer.token_to_id(SpecialToken.PAD)
    pad_token_id = src_tokenizer.token_to_id(SpecialToken.PAD)

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
    train_dataset = BilingualDataset(
        raw_dataset['train'],
        src_tokenizer,
        target_tokenizer,
        args.src_seq_length,
        args.target_seq_length,
        source_key='source',
        target_key='target',
        add_padding_tokens=False,
        include_source_target_text=False,
    )
    test_dataset = BilingualDataset(
        raw_dataset['test'],
        src_tokenizer,
        target_tokenizer,
        args.src_seq_length,
        args.target_seq_length,
        source_key='source',
        target_key='target',
        add_padding_tokens=False,
        include_source_target_text=False,
    )
    pad_features = ['input_ids', 'labels']
    data_collator = CollatorWithPadding(pad_token_id, pad_features)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=data_collator,
    )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    use_fp16 = args.fp16 and device.type == 'cuda'

    checkpoint_states = None
    pretrained_checkpoint_states = None
    if args.from_checkpoint is not None:
        # resume from previous checkpoint
        print(f'Loading states from checkpoint {args.from_checkpoint}')

        checkpoint_states = torch.load(args.from_checkpoint, map_location=device)
        required_keys = ['model', 'optimizer', 'lr_scheduler', 'config']
        if use_fp16:
            required_keys.append('scaler')
        for key in required_keys:
            if key not in checkpoint_states:
                raise ValueError(f'Missing key "{key}" in checkpoint {args.from_checkpoint}')
        bart_for_nmt_config = checkpoint_states['config']
        assert isinstance(bart_for_nmt_config, BartForNMTConfig)
        bart_for_nmt_config.device = device
    else:
        # start training from scratch or from pre-trained model

        # TODO: currently, the code below assumes `target_tokenizer`
        # is the same as the one that used during pre-training.
        bart_for_nmt_config = BartForNMTConfig(
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
            foreign_encoder_num_layers=args.foreign_encoder_num_layers,
            foreign_encoder_num_heads=args.foreign_encoder_num_heads,
        )
        if args.from_pretrained is not None:
            print(f'Starting fine-tuning from pretrained checkpoint {args.from_pretrained}')
            pretrained_checkpoint_states = torch.load(args.from_pretrained, map_location=device)
            required_keys = ['model', 'config']
            for key in required_keys:
                if key not in pretrained_checkpoint_states:
                    raise ValueError(f'Missing key "{key}" in checkpoint {args.from_pretrained}')
            pretrained_bart_config = pretrained_checkpoint_states['config']
            assert isinstance(pretrained_bart_config, BartForNMTConfig)
            pretrained_config_keys = [
                'target_pad_token_id', 'target_start_token_id', 'target_end_token_id',
                'target_vocab_size', 'hidden_size', 'intermediate_size',
                'encoder_num_heads', 'encoder_num_hidden_layers', 'decoder_num_heads',
                'decoder_num_hidden_layers',
            ]
            for key in pretrained_config_keys:
                value = getattr(pretrained_bart_config, key)
                setattr(bart_for_nmt_config, key, value)
        else:
            print('Starting training from scratch')

    # model, optimizer, lr_scheduler, scaler
    model = BartForNMT(bart_for_nmt_config)
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
    if pretrained_checkpoint_states is not None:
        model.load_state_dict(pretrained_checkpoint_states['model'], strict=False)
    if checkpoint_states is not None:
        model.load_state_dict(checkpoint_states['model'])
        optimizer.load_state_dict(checkpoint_states['optimizer'])
        lr_scheduler.load_state_dict(checkpoint_states['lr_scheduler'])
        scaler.load_state_dict(checkpoint_states['scaler'])
        if 'global_step' in checkpoint_states:
            initial_global_step = checkpoint_states['global_step']
        if 'accum_train_loss' in checkpoint_states:
            initial_accum_train_loss = checkpoint_states['accum_train_loss']

    # freezing params for fine-tuning on machine translation task
    # for more details, see https://arxiv.org/abs/1910.13461 (Section 3.4)
    if args.freeze_params:
        model.freeze_params()
    else:
        model.unfreeze_params()

    # tensorboard
    writer = SummaryWriter(args.expr_dir)

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

    # trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        src_tokenizer=src_tokenizer,
        target_tokenizer=target_tokenizer,
        args=training_args,
        bart_config=bart_for_nmt_config,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        writer=writer,
    )
    print(f'Model has {model.num_params()} parameters')
    trainer.train(train_data_loader, test_data_loader)

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tuning BART for neural machine translation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.fine_tune_nmt_opts(parser)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    train_model(args)


if __name__ == '__main__':
    main()
