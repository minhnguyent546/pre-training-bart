from typing import Any

from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from tokenizers import Tokenizer

from bart.constants import SpecialToken


class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        src_seq_length: int,
        target_seq_length: int,
        source_key: str = 'source',
        target_key: str = 'target',
        add_padding_tokens: bool = False,
        include_source_target_text: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_key = source_key
        self.target_key = target_key
        self.src_seq_length = src_seq_length
        self.target_seq_length = target_seq_length
        self.add_padding_tokens = add_padding_tokens
        self.include_source_target_text = include_source_target_text

        assert src_tokenizer.token_to_id(SpecialToken.SOS) == target_tokenizer.token_to_id(SpecialToken.SOS)
        assert src_tokenizer.token_to_id(SpecialToken.EOS) == target_tokenizer.token_to_id(SpecialToken.EOS)
        assert src_tokenizer.token_to_id(SpecialToken.PAD) == target_tokenizer.token_to_id(SpecialToken.PAD)

        self.sos_token_id = src_tokenizer.token_to_id(SpecialToken.SOS)
        self.eos_token_id = src_tokenizer.token_to_id(SpecialToken.EOS)
        self.pad_token_id = src_tokenizer.token_to_id(SpecialToken.PAD)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_text = self.dataset[index][self.source_key]
        target_text = self.dataset[index][self.target_key]

        encoder_input_ids = self.src_tokenizer.encode(src_text).ids
        labels = self.target_tokenizer.encode(target_text).ids

        encode_num_paddings, decode_num_paddings = 0, 0
        if self.add_padding_tokens:
            encode_num_paddings = self.src_seq_length - len(encoder_input_ids) - 2  # exclude <SOS> & <EOS>
            decode_num_paddings = self.target_seq_length - len(labels) - 1  # exclude <SOS> | <EOS>

        assert encode_num_paddings >= 0, "The length of the source text is too long"
        assert decode_num_paddings >= 0, "The length of the target text is too long"

        encoder_input_ids = torch.cat([
            Tensor([self.sos_token_id]),
            Tensor(encoder_input_ids),
            Tensor([self.eos_token_id]),
            Tensor([self.pad_token_id]).repeat(encode_num_paddings)
        ]).type(torch.int32)
        decoder_input_ids = torch.cat([
            Tensor([self.sos_token_id]),
            Tensor(labels),
            Tensor([self.pad_token_id]).repeat(decode_num_paddings)
        ]).type(torch.int32)
        labels = torch.cat([
            Tensor(labels),
            Tensor([self.eos_token_id]),
            Tensor([self.pad_token_id]).repeat(decode_num_paddings)
        ]).type(torch.int64)  # int32 has a problem with nll loss forward on cuda

        if self.add_padding_tokens:
            assert encoder_input_ids.size(0) == self.src_seq_length
            assert decoder_input_ids.size(0) == self.target_seq_length
            assert labels.size(0) == self.target_seq_length

        item = {
            'input_ids': encoder_input_ids,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }
        if self.include_source_target_text:
            item['source_text'] = src_text
            item['target_text'] = target_text
        return item

class CollatorWithPadding:
    def __init__(self, padding_value: int, pad_features: list[str] | None = None) -> None:
        self.padding_value = padding_value
        self.pad_features = pad_features if pad_features is not None else []

    def __call__(self, original_batch: list[dict[str, Any]]) -> dict[str, Any]:
        all_features = original_batch[0].keys()

        # list of feature that will not be padded
        remain_features = [key for key in all_features if key not in self.pad_features]

        feature_dict = {key: [item[key] for item in original_batch] for key in self.pad_features}
        batch = {key: [item[key] for item in original_batch] for key in remain_features}
        feature_dict = {
            key: pad_sequence(value, batch_first=True, padding_value=self.padding_value)
            for key, value in feature_dict.items()
        }
        batch.update(feature_dict)
        return batch
