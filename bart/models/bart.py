"""BART model"""

from dataclasses import dataclass

from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .transformer import Transformer, TransformerBase, TransformerConfig
from . import utils


@dataclass
class BartConfig(TransformerConfig):
    pooler_dropout: float = 0.1
    pooler_activation: str = 'tanh'

class BartBase(TransformerBase):
    pass

class Bart(Transformer):
    def __init__(self, config: BartConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attn_mask: Tensor | None = None,
        decoder_attn_mask: Tensor | None = None
    ) -> Tensor:
        decoder_output = super().forward(input_ids, decoder_input_ids, attn_mask, decoder_attn_mask)
        return decoder_output

class BartClassificationHead(nn.Module):
    """A head for sentence-level classification tasks."""
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        pooler_activation: str,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation_fn = utils.get_activation(pooler_activation)
        self.dropout = nn.Dropout(pooler_dropout)

        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, sentence_repr_states: Tensor) -> Tensor:
        # standard pooler head actions
        output = self.dropout(sentence_repr_states)
        output = self.dense(output)
        output = self.activation_fn(output)
        output = self.dropout(output)

        # project to the number of classes
        output = self.proj(output)
        return output

class BartForGeneration(BartBase):
    """Bart for text generation tasks (really like transformer).

    Unlike Bert model, Bart does not use additional feed-forward network before
    word prediction.
    """
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.model = Bart(config)
        if not self.config.shared_vocab and config.target_vocab_size is None:
            raise ValueError('`target_vocab_size` must be provided if `shared_vocab` is `False`')
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.target_vocab_size or config.src_vocab_size,
        )

        if self.config.tie_weights:
            self._tie_weights()
        self.post_init()

    def _tie_weights(self):
        self.lm_head.weight = self.model.target_token_embeddings.weight

    def _init_module_weights_fn(self, module, std: float = 0.02):
        """Following the same initialization as in BERT."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight.data[module.padding_idx])

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attn_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if input_mask is None:
            input_mask = input_ids != self.config.src_pad_token_id

        if decoder_input_ids is None and labels is None:
            raise ValueError('labels must be provided if decoder_input_ids is not provided')
        # if decoder_input_ids is not provided, it will be inferred from input_ids
        if decoder_input_ids is None:
            assert labels is not None
            decoder_input_ids = utils.shift_tokens_right(labels, self.config.target_start_token_id)
            decoder_input_ids[decoder_input_ids == self.config.target_end_token_id] = self.config.target_pad_token_id
        if decoder_attn_mask is None:
            decoder_attn_mask = decoder_input_ids != self.config.target_pad_token_id

        decoder_output = self.model(
            input_ids,
            decoder_input_ids,
            attn_mask=input_mask,
            decoder_attn_mask=decoder_attn_mask,
        )  # (batch_size, seq_length, hidden_size)
        lm_logits = self.lm_head(decoder_output)
        lm_loss = self.get_lm_loss(lm_logits, labels)

        return lm_logits, lm_loss

    def get_lm_loss(self, lm_logits: Tensor, labels: Tensor | None = None) -> Tensor | None:
        if labels is None:
            return None
        lm_loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.config.target_pad_token_id,
        )
        return lm_loss
