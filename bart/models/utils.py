from torch import Tensor, nn
import torch
import torch.nn.functional as Fun

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

def create_encoder_4d_attn_mask(input_ids: Tensor, attn_mask: Tensor) -> Tensor:
    if attn_mask.shape != input_ids.shape:
        raise ValueError(
            'Expected input_ids and attn_mask have the same shape, '
            f'got {input_ids.shape} and {attn_mask.shape}'
        )
    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
    return attn_mask

def create_decoder_4d_attn_causal_mask(
    decoder_input_ids: Tensor,
    decoder_attn_mask: Tensor,
) -> Tensor:
    if decoder_attn_mask.shape != decoder_input_ids.shape:
        raise ValueError(
            'Expected decoder_input_ids and decoder_attn_mask have the same shape, '
            f'got {decoder_input_ids.shape} and {decoder_attn_mask.shape}'
        )
    seq_length = decoder_input_ids.shape[1]
    causal_mask = create_causal_mask(seq_length).to(decoder_attn_mask.device)
    decoder_attn_mask = decoder_attn_mask.unsqueeze(1).unsqueeze(2)
    decoder_attn_causal_mask = decoder_attn_mask & causal_mask
    return decoder_attn_causal_mask

def create_causal_mask(seq_length: int) -> Tensor:
    return torch.tril(torch.ones(seq_length, seq_length)).bool()

@torch.no_grad()
def greedy_search_decode(
    model,
    device: torch.device,
    encoder_input_ids: Tensor,
    target_tokenizer: Tokenizer,
    max_seq_length: int,
    encoder_attn_mask: Tensor | None = None,
) -> Tensor:
    """
    Args:
        model: model to be used for decoding
        device (torch.device): training device
        encoder_input_ids (Tensor): encoder input ids, shape ``(src_seq_length,)``
        target_tokenizer (Tokenizer): target tokenizer
        max_seq_length (int): maximum sequence length
        encoder_attn_mask (Tensor | None): attention mask for encoder input, shape ``(src_seq_length,)`` (default: None)

    Returns:
        Tensor: tensor of predicted token ids
    """

    sos_token_id = target_tokenizer.token_to_id(SpecialToken.SOS)
    eos_token_id = target_tokenizer.token_to_id(SpecialToken.EOS)

    encoder_input_ids = encoder_input_ids.unsqueeze(0).to(device)
    encoder_output = None
    causal_mask = create_causal_mask(max_seq_length).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, max_seq_length, max_seq_length)

    # initialize decoder input which contains only [SOS] token
    decoder_input_ids = torch.empty((1, 1)).fill_(sos_token_id).type_as(encoder_input_ids).to(device)
    for _ in range(max_seq_length):
        # create mask for decoder input
        decoder_attn_mask = causal_mask[..., :decoder_input_ids.size(1), :decoder_input_ids.size(1)]

        outputs = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_output=encoder_output,
        )
        if encoder_output is None:
            encoder_output = outputs.encoder_output

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
    target_tokenizer: Tokenizer,
    max_seq_length: int,
    return_topk: int = 1,
    encoder_attn_mask: Tensor | None = None,
) -> list[Tensor]:
    """
    Args:
        model (Transformer): model to be used for decoding
        device (torch.device): device
        beam_size (int): beam size
        encoder_input_ids (Tensor): encoder input ids, shape ``(src_seq_length,)``
        target_tokenizer (Tokenizer): target tokenizer
        max_seq_length (int): maximum sequence length
        return_topk (int): return top k best candidates (default: 1)
        encoder_attn_mask (Tensor | None): attention mask for encoder input, shape ``(src_seq_length,)`` (default: None)

    Returns:
        list[Tensor]: list of candidate tensors of predicted token ids
    """

    sos_token_id = target_tokenizer.token_to_id(SpecialToken.SOS)
    eos_token_id = target_tokenizer.token_to_id(SpecialToken.EOS)

    encoder_input_ids = encoder_input_ids.unsqueeze(0).to(device)
    encoder_output = None
    causal_mask = create_causal_mask(max_seq_length).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, max_seq_length, max_seq_length)

    # initialize decoder input which contains only [SOS] token
    decoder_input_ids = torch.empty((1, 1)).fill_(sos_token_id).type_as(encoder_input_ids).to(device)

    # candidate list of tuples (decoder_input, log_score)
    cands = [(decoder_input_ids, 0.0)]
    for _ in range(max_seq_length):
        new_cands = []

        for cand, log_score in cands:
            # do not expand the candidate that have reached <EOS> token
            if cand[0, -1].item() == eos_token_id:
                new_cands.append((cand, log_score))
                continue

            # create mask for decoder input
            cand_mask = causal_mask[..., :cand.size(1), :cand.size(1)]

            outputs = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=cand,
                encoder_attn_mask=encoder_attn_mask,
                decoder_attn_mask=cand_mask,
                encoder_output=encoder_output,
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
                new_cands.append((new_cand, log_score + token_prob))

        cands = sorted(new_cands, key=lambda x: x[1], reverse=True)
        cands = cands[:beam_size]

        if all([cand[0][-1].item() == eos_token_id for cand, _ in cands]):
            break

    assert len(cands) == beam_size
    cands = cands[:return_topk]
    result_cands = [cand[0].squeeze(0) for cand in cands]
    return result_cands
