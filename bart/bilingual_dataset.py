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

        self.post_init()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source_text = self.dataset[index][self.source_key]
        target_text = self.dataset[index][self.target_key]

        input_ids = self.src_tokenizer.encode(source_text).ids
        labels = self.target_tokenizer.encode(target_text).ids

        input_num_paddings, label_num_paddings = 0, 0
        if self.add_padding_tokens:
            input_num_paddings = self.src_seq_length - len(input_ids) - 2  # exclude <SOS> & <EOS>
            label_num_paddings = self.target_seq_length - len(labels) - 1  # exclude <SOS> | <EOS>

        assert input_num_paddings >= 0, "The length of the source text is too long"
        assert label_num_paddings >= 0, "The length of the target text is too long"

        input_ids = torch.cat([
            Tensor([self.sos_token_id]),
            Tensor(input_ids),
            Tensor([self.eos_token_id]),
            Tensor([self.pad_token_id]).repeat(input_num_paddings)
        ]).type(torch.int32)
        labels = torch.cat([
            Tensor(labels),
            Tensor([self.eos_token_id]),
            Tensor([self.pad_token_id]).repeat(label_num_paddings)
        ]).type(torch.int64)  # labels must be of type int64 for NLLLoss to work

        if self.add_padding_tokens:
            assert input_ids.size(0) == self.src_seq_length
            assert labels.size(0) == self.target_seq_length

        item = {
            'input_ids': input_ids,
            'labels': labels,
        }
        if self.include_source_target_text:
            item['source_text'] = source_text
            item['target_text'] = target_text
        return item

    def post_init(self) -> None:
        self.remove_invalid_pairs()

    def remove_invalid_pairs(self) -> None:
        def _remove_invalid_pair_fn(examples: dict[str, list]) -> list[bool]:
            is_valid_item = [True] * len(examples[self.source_key])
            for item_id, (source_text, target_text) in enumerate(zip(examples[self.source_key], examples[self.target_key])):
                if min(len(source_text), len(target_text)) == 0:
                    is_valid_item[item_id] = False
                    continue
                len_ratio = max(len(source_text), len(target_text)) / min(len(source_text), len(target_text))
                if len_ratio > 1.5:
                    is_valid_item[item_id] = False
                    continue
                num_src_tokens = len(self.src_tokenizer.encode(source_text).tokens)
                num_target_tokens = len(self.target_tokenizer.encode(target_text).tokens)
                is_valid_item[item_id] = (min(num_src_tokens, num_target_tokens) > 0 and
                                          num_src_tokens + 2 <= self.src_seq_length and
                                          num_target_tokens + 1 <= self.target_seq_length)
            return is_valid_item
        self.dataset = self.dataset.filter(_remove_invalid_pair_fn, batched=True)

class CollatorWithPadding:
    def __init__(self, padding_value: int, pad_features: list[str] | None = None) -> None:
        self.padding_value = padding_value
        self.pad_features = pad_features if pad_features is not None else []

    def __call__(self, original_batch: list[dict[str, Any]]) -> dict[str, Any]:
        # assume each item in `original_batch` have the same keys/features
        all_features = original_batch[0].keys()

        # list of features that will be padded
        pad_features = [feature for feature in self.pad_features if feature in all_features]
        # list of features that will not be padded
        no_pad_features = [feature for feature in all_features if feature not in pad_features]

        # list[dict[str, Any]] -> dict[str, Any]
        feature_dict = {
            feature: [item[feature] for item in original_batch]
            for feature in pad_features
        }
        feature_dict = {
            feature: pad_sequence(value, batch_first=True, padding_value=self.padding_value)
            for feature, value in feature_dict.items()
        }
        batch = {
            feature: [item[feature] for item in original_batch]
            for feature in no_pad_features
        }
        batch.update(feature_dict)
        return batch
