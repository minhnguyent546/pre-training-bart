import glob
from operator import itemgetter
import os

from tqdm.autonotebook import tqdm

from torch import Tensor, nn
import torch
import torch.nn.functional as Fun
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from bart.constants import SpecialToken


def get_activation(act_type: str) -> nn.Module:
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Unrecognized activation type: {act_type}.')

def shift_tokens_right(tokens: Tensor, start_token_id: int) -> Tensor:
    if tokens.dim() != 2:
        raise ValueError(f'Expected tokens tensor is a 2D tensor, but got a tensor with shape: {tokens.size()}')

    shifted = tokens.new_zeros(tokens.shape)
    shifted[:, :1] = start_token_id
    shifted[:, 1:] = tokens[:, :-1].detach().clone()
    return shifted

def initialize_bert_params_fn(module, std: float = 0.02):
    """Following the same initialization as in BERT."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight.data, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, mean=0.0, std=std)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight.data[module.padding_idx])

def freeze_parameters(model: nn.Module, exclude: list[str] | None = None) -> list[str]:
    if exclude is None:
        exclude = []
    freezed_params: list[str] = []
    for name, param in model.named_parameters():
        if not any(name.startswith(e) for e in exclude):
            param.requires_grad = False
            freezed_params.append(name)
    return freezed_params

def unfreeze_parameters(model: nn.Module, freezed_params: list[str]) -> None:
    for name, param in model.named_parameters():
        if name in freezed_params:
            param.requires_grad = True

def create_encoder_4d_attn_mask(
    attn_mask: Tensor,
    input_shape: torch.Size | tuple[int, ...] | list[int],
    target_seq_length: int | None = None,
) -> Tensor:
    batch_size, src_seq_length = input_shape
    if target_seq_length is None:
        target_seq_length = src_seq_length
    if attn_mask.dim() == 4:
        expected_shape = (batch_size, 1, target_seq_length, src_seq_length)
        if tuple(attn_mask.shape) != expected_shape:
            raise ValueError(
                f'Expected `attn_mask` has shape {expected_shape}, '
                f'but got {tuple(attn_mask.shape)}'
            )
    elif attn_mask.dim() == 2:
        if tuple(attn_mask.shape) != input_shape:
            raise ValueError(
                f'`attn_mask` must have shape {input_shape} in order to expand it to 4D, '
                f'but got {tuple(attn_mask.shape)}'
            )
        # (batch_size, src_seq_length) -> (batch_size, 1, target_seq_length, src_seq_length)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_length)
        attn_mask = attn_mask.expand(batch_size, 1, target_seq_length, src_seq_length)
    else:
        raise ValueError(
            'Expected `attn_mask` to have 2 or 4 dimensions, '
            f'but got a tensor with shape: {tuple(attn_mask.shape)}'
        )
    return attn_mask

def create_decoder_4d_causal_attn_mask(
    attn_mask: Tensor | None,
    input_shape: torch.Size | tuple[int, ...] | list[int],
    device: torch.device = torch.device('cpu'),
    cached_kv_length: int = 0,
) -> Tensor:
    batch_size, query_length = input_shape
    kv_length = query_length + cached_kv_length
    if attn_mask is not None:
        if attn_mask.dim() == 4:
            expected_shape = (batch_size, 1, query_length, kv_length)
            if tuple(attn_mask.shape) != expected_shape:
                raise ValueError(
                    f'Expected `attn_mask` has shape {expected_shape}, '
                    f'but got {tuple(attn_mask.shape)}'
                )
        elif attn_mask.dim() == 2:
            if cached_kv_length > 0:
                attn_mask = torch.cat([
                    torch.ones((batch_size, cached_kv_length), dtype=torch.bool, device=device),
                    attn_mask,
                ], dim=-1)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, kv_length)
            attn_mask = attn_mask.expand(batch_size, 1, query_length, kv_length)  # (batch_size, 1, query_length, kv_length)
        else:
            raise ValueError(
                'Expected `attn_mask` to have 2 or 4 dimensions, '
                f'but got a tensor with shape: {tuple(attn_mask.shape)}'
            )
    else:
        attn_mask = create_causal_mask(query_length).to(device)  # (query_length, query_length)
        if cached_kv_length > 0:
            attn_mask = torch.cat([
                torch.ones((query_length, cached_kv_length), dtype=torch.bool, device=device),
                attn_mask,
            ], dim=-1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, query_length, kv_length)
        attn_mask = attn_mask.expand(batch_size, 1, query_length, kv_length)  # (batch_size, 1, query_length, kv_length)
    return attn_mask

def create_causal_mask(seq_length: int) -> Tensor:
    return torch.tril(torch.ones(seq_length, seq_length)).bool()

def eval_model(
    model,
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
            labels = batch['labels'].to(device).type(torch.int64)
            decoder_input_ids = None
            decoder_input_mask = None
            if 'decoder_input_ids' in batch:
                decoder_input_ids = batch['decoder_input_ids'].to(device).type(torch.int32)

            if 'input_mask' in batch:
                input_mask = batch['input_mask'].to(device).type(torch.bool)
            else:
                input_mask = input_ids != model.config.src_pad_token_id

            if 'decoder_input_mask' in batch:
                decoder_input_mask = batch['decoder_input_mask'].to(device).type(torch.bool)
            elif decoder_input_ids is not None:
                decoder_input_mask = decoder_input_ids != model.config.target_pad_token_id

            outputs = model(
                encoder_input_ids=input_ids,
                encoder_attn_mask=input_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attn_mask=decoder_input_mask,
                labels=labels,
            )
            loss = outputs.lm_loss
            accum_valid_loss += loss.item()

            batch_iter.set_postfix({'loss': f'{loss.item():0.3f}'})

    model.train(is_training)

    num_iterations = len(eval_data_loader)
    return {
        'loss': accum_valid_loss / num_iterations,
    }

@torch.no_grad()
def greedy_search_decode(
    model,
    device: torch.device,
    encoder_input_ids: Tensor,
    max_seq_length: int,
    encoder_attn_mask: Tensor | None = None,
    use_cache: bool = False,
) -> Tensor:
    """
    Args:
        model: model to be used for decoding
        device (torch.device): training device
        encoder_input_ids (Tensor): encoder input ids, shape ``(src_seq_length,)``
        max_seq_length (int): maximum sequence length
        encoder_attn_mask (Tensor | None): attention mask for encoder input, shape ``(src_seq_length,)`` (default: None)
        use_cache (bool): Whether to use kv cache during decoding (defalt: False)

    Returns:
        Tensor: tensor of predicted token ids
    """
    sos_token_id = model.config.target_start_token_id
    eos_token_id = model.config.target_end_token_id

    encoder_input_ids = encoder_input_ids.unsqueeze(0).to(device)
    if encoder_attn_mask is not None:
        encoder_attn_mask = encoder_attn_mask.unsqueeze(0).to(device)

    encoder_output = None
    kv_caches = None

    # initialize decoder input which contains only [SOS] token
    decoder_input_ids = torch.empty((1, 1)).fill_(sos_token_id).type_as(encoder_input_ids).to(device)
    for _ in range(max_seq_length):
        cur_input_ids = decoder_input_ids[:, [-1]] if use_cache else decoder_input_ids
        outputs = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=cur_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            encoder_output=encoder_output,
            kv_caches=kv_caches,
            use_cache=use_cache,
        )
        if encoder_output is None:
            encoder_output = outputs.encoder_output
        kv_caches = outputs.kv_caches

        # get token with highest probability
        logits = outputs.lm_logits
        logits = logits[:, -1, :]  # (1, target_vocab_size)
        next_token = logits.argmax(dim=-1)  # (1,)

        # concatenate the next token to the decoder input for the next prediction
        decoder_input_ids = torch.cat([
            decoder_input_ids,
            torch.empty((1, 1)).fill_(next_token.item()).type_as(decoder_input_ids).to(device)
        ], dim=1)

        # if we reach the <EOS> token, then stop
        if next_token == eos_token_id:
            break

    return decoder_input_ids.squeeze(0)

def length_penalty(length: int, alpha: float = 0.6) -> float:
    """
    As formula described in Wu et al. (2016)
    """
    return (5 + length) ** alpha / (5 + 1) ** alpha

@torch.no_grad()
def beam_search_decode(
    model,
    device: torch.device,
    beam_size: int,
    encoder_input_ids: Tensor,
    max_seq_length: int,
    return_topk: int = 1,
    encoder_attn_mask: Tensor | None = None,
    use_cache: bool = False,
) -> list[Tensor]:
    """
    Args:
        model (Transformer): model to be used for decoding
        device (torch.device): device
        beam_size (int): beam size
        encoder_input_ids (Tensor): encoder input ids, shape ``(src_seq_length,)``
        max_seq_length (int): maximum sequence length
        return_topk (int): return top k best candidates (default: 1)
        encoder_attn_mask (Tensor | None): attention mask for encoder input, shape ``(src_seq_length,)`` (default: None)
        use_cache (bool): whether to use kv cache during decoding (default: False)

    Returns:
        list[Tensor]: list of candidate tensors of predicted token ids
    """
    encoder_input_ids = encoder_input_ids.unsqueeze(0).to(device)
    if encoder_attn_mask is not None:
        encoder_attn_mask = encoder_attn_mask.unsqueeze(0).to(device)

    encoder_output = None
    sos_token_id = model.config.target_start_token_id
    eos_token_id = model.config.target_end_token_id

    # initialize decoder input which contains only [SOS] token
    decoder_input_ids = torch.empty((1, 1)).fill_(sos_token_id).type_as(encoder_input_ids).to(device)

    # candidate list of tuples (decoder_input, log_score, kv_caches)
    cands = [(decoder_input_ids, 0.0, None)]
    for _ in range(max_seq_length):
        new_cands = []

        for cand, log_score, kv_caches in cands:
            # do not expand the candidate that have reached <EOS> token
            if cand[0, -1].item() == eos_token_id:
                new_cands.append((cand, log_score, kv_caches))
                continue

            cur_input_ids = cand if not use_cache else cand[:, [-1]]
            outputs = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=cur_input_ids,
                encoder_attn_mask=encoder_attn_mask,
                encoder_output=encoder_output,
                kv_caches=kv_caches,
                use_cache=use_cache,
            )
            if encoder_output is None:
                encoder_output = outputs.encoder_output

            # get next token probabilities
            # logits: shape ``(1, target_vocab_size)``
            # topk_prob       : shape ``(1, beam_size)``
            # topk_token      : shape ``(1, beam_size)``
            logits = outputs.lm_logits
            logits = logits[:, -1, :]  # (1, target_vocab_size)

            output = Fun.log_softmax(logits, dim=-1) / length_penalty(cand.size(1) + 1)
            # get the top k largest tokens
            topk_token_prob, topk_token = torch.topk(output, beam_size, dim=1)
            for j in range(beam_size):
                # token: shape ``(1, 1)``
                # token_prob: scalar
                token = topk_token[0][j].unsqueeze(0).unsqueeze(0)
                token_prob = topk_token_prob[0][j].item()

                new_cand = torch.cat([cand, token], dim=1)
                new_cands.append((new_cand, log_score + token_prob, outputs.kv_caches))

        cands = sorted(new_cands, key=itemgetter(1), reverse=True)
        cands = cands[:beam_size]

        if all([cand[0][-1].item() == eos_token_id for cand, _, __ in cands]):
            break

    assert len(cands) == beam_size
    cands = cands[:return_topk]
    result_cands = [cand[0].squeeze(0) for cand in cands]
    return result_cands

def ensure_num_saved_checkpoints(
    checkpoints_dir: str,
    model_basename: str,
    limit: int,
) -> None:
    checkpoints = glob.glob(os.path.join(checkpoints_dir, f'{model_basename}-*.pt'))
    checkpoints = list(checkpoints)
    if len(checkpoints) <= limit:
        return

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1][:-3]))
    for cp in checkpoints[:-limit]:
        os.remove(cp)
