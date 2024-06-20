from tqdm.autonotebook import tqdm

import torch
from torch import Tensor

import datasets
import evaluate
from tokenizers import Tokenizer

import bart.models.utils as model_utils
from bart.constants import LOWER_ONE_EIGHTH_BLOCK, SpecialToken
from bart.models import BartBase


def compute_dataset_bleu(
    model: BartBase,
    dataset: datasets.Dataset,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    seq_length: int,
    beam_size: int | None = None,
    beam_return_topk: int = 1,
    log_sentences: bool = False,
    logging_interval: int = 20,
    max_steps: int | None = None,
) -> float:

    device = model.device
    target_text_list = []
    pred_text_list = []

    total_steps = len(dataset)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    dataset_iterator = tqdm(
        enumerate(dataset),
        desc='Computing validation BLEU',
        total=total_steps,
    )
    cand_list = None
    cand_text_list = None

    # set model in evaluation mode
    is_training = model.training
    model.eval()

    sacrebleu = evaluate.load('sacrebleu')
    ignored_tokens = [SpecialToken.SOS, SpecialToken.EOS, SpecialToken.PAD, SpecialToken.UNK]

    with torch.no_grad():
        for item_idx, item in dataset_iterator:
            if item_idx >= total_steps:
                break

            # input_ids has form of: <s> ... </s> [PAD] [PAD]...
            # labels has form of   : ... </s> [PAD] [PAD]...
            input_ids = Tensor(item['input_ids']).type(torch.int32)
            labels = Tensor(item['labels']).type(torch.int64)
            input_mask = None
            if 'input_mask' in item:
                input_mask = Tensor(item['input_mask']).type(torch.int32)

            # retrieving source and target text
            if 'source_text' in item:
                source_text = item['source_text']
            else:
                if 'source_tokens' in item:
                    source_tokens = item['source_tokens']
                    source_token_ids = [
                        src_tokenizer.token_to_id(token)
                        for token in source_tokens
                        if token not in ignored_tokens
                    ]
                else:
                    source_token_ids = [
                        token_id for token_id in input_ids
                        if src_tokenizer.id_to_token(token_id) not in ignored_tokens
                    ]
                source_text = src_tokenizer.decode(source_token_ids, skip_special_tokens=False)
            if 'target_text' in item:
                target_text = item['target_text']
            else:
                if 'target_tokens' in item:
                    target_tokens = item['target_tokens']
                    target_token_ids = [
                        target_tokenizer.token_to_id(token)
                        for token in target_tokens
                        if token not in ignored_tokens
                    ]
                else:
                    target_token_ids = [
                        token_id for token_id in labels
                        if target_tokenizer.id_to_token(token_id) not in ignored_tokens
                    ]
                target_text = target_tokenizer.decode(target_token_ids, skip_special_tokens=False)

            if beam_size is not None and beam_size > 1:
                # decoding with beam search
                cand_list = model_utils.beam_search_decode(
                    model,
                    device,
                    beam_size,
                    input_ids,
                    target_tokenizer,
                    seq_length,
                    return_topk=beam_return_topk,
                    encoder_attn_mask=input_mask,
                )
                cand_list = [cand.detach().cpu().numpy() for cand in cand_list]
                cand_text_list = []
                for cand in cand_list:
                    cand_text = target_tokenizer.decode(cand).replace('_', LOWER_ONE_EIGHTH_BLOCK)
                    cand_text_list.append(cand_text)
                pred_token_ids = cand_list[0]
            else:
                # decoding with greedy search
                pred_token_ids = model_utils.greedy_search_decode(
                    model,
                    device,
                    input_ids,
                    target_tokenizer,
                    seq_length,
                    input_mask,
                )
                pred_token_ids = pred_token_ids.detach().cpu().numpy()

            # tokenizer.decode method will remove special tokens by default (e.g. <UNK>)
            # it should be, because keep <UNK> tokens will increase the BLEU score
            # but has no meaning. See Post, 2018
            pred_text = target_tokenizer.decode(pred_token_ids)

            source_text = source_text.replace('_', LOWER_ONE_EIGHTH_BLOCK)
            target_text = target_text.replace('_', LOWER_ONE_EIGHTH_BLOCK)
            pred_text = pred_text.replace('_', LOWER_ONE_EIGHTH_BLOCK)

            target_text_list.append(target_text)
            pred_text_list.append(pred_text)

            if log_sentences and item_idx % logging_interval == 0:
                bleu_score = sacrebleu.compute(predictions=[pred_text], references=[target_text])

                dataset_iterator.write(f'Source: {source_text}')
                dataset_iterator.write(f'Target: {target_text}')
                if cand_text_list is not None:
                    for cand_text_idx, cand_text in enumerate(cand_text_list):
                        dataset_iterator.write(f'Predicted-{cand_text_idx + 1}: {cand_text}')
                else:
                    dataset_iterator.write(f'Predicted: {pred_text}')

                dataset_iterator.write(f'BLEU: {bleu_score["score"]:0.3f}')

    dataset_blue_score = sacrebleu.compute(
        predictions=pred_text_list,
        references=target_text_list,
    )

    # set model back to training mode
    model.train(is_training)

    return dataset_blue_score['score']
