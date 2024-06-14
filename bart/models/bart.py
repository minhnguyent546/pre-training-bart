"""BART model"""

import math
from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor
from torch import nn

import bart.models.utils as model_utils
from bart.models.transformer import (
    InputEmbeddings,
    LayerNormalization,
    PositionEmbeddings,
    Transformer,
    TransformerBase,
    TransformerConfig,
    TransformerEncoderLayer,
    TransformerOutput,
)
from bart.models.utils import initialize_bert_params_fn


@dataclass
class BartConfig(TransformerConfig):
    pooler_dropout: float = 0.1
    pooler_activation: str = 'tanh'

@dataclass
class BartForNMTConfig(BartConfig):
    foreign_encoder_num_layers: int = 6
    foreign_encoder_num_heads: int = 8

@dataclass
class BartForGenerationOutput:
    encoder_output: Tensor
    decoder_output: Tensor | None = None
    lm_logits: Tensor | None = None
    lm_loss: Tensor | None = None

class BartBase(TransformerBase):
    pass

class Bart(Transformer):
    def __init__(self, config: BartConfig):
        super().__init__(config)

    def forward(
        self,
        encoder_input_ids: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        encoder_attn_mask: Tensor | None = None,
        decoder_attn_mask: Tensor | None = None,
        encoder_output: Tensor | None = None,
    ) -> TransformerOutput:
        outputs = super().forward(
            encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_output=encoder_output,
        )
        return outputs

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
        self.activation_fn = model_utils.get_activation(pooler_activation)
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

class BertEncoderForForeignLanguage(nn.Module):
    """A small additional encoder that replaces the input embeddings in BERT."""
    def __init__(self, config: BartForNMTConfig):
        super().__init__()
        self.token_embeddings = InputEmbeddings(
            config.src_vocab_size,
            config.hidden_size,
            scale_factor=math.sqrt(config.hidden_size),
        )
        self.position_embeddings = PositionEmbeddings(config.hidden_size, config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.hidden_size,
                config.foreign_encoder_num_heads,
                config.intermediate_size,
                config.activation,
                pre_norm=config.pre_norm,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
            )
            for _ in range(config.foreign_encoder_num_layers)
        ])
        self.pre_norm = config.pre_norm
        if self.pre_norm:
            self.layer_norm = LayerNormalization(config.hidden_size)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        # embed tokens and positions
        x = self.token_embeddings(x) + self.position_embeddings(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        if self.pre_norm:
            x = self.layer_norm(x)
        return x

class BartForGeneration(BartBase):
    """Bart for text generation tasks (really like transformer).

    Unlike Bert model, Bart does not use additional feed-forward network before
    word prediction.
    """
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.device = config.device
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
        assert self.lm_head.weight.shape == self.model.decoder.token_embeddings.weight.shape
        self.lm_head.weight = self.model.decoder.token_embeddings.weight

    def _init_model_weights(self, std: float = 0.02) -> None:
        self.apply(lambda module: initialize_bert_params_fn(module, std=std))

    def forward(
        self,
        encoder_input_ids: Tensor | None = None,
        encoder_attn_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attn_mask: Tensor | None = None,
        encoder_output: Tensor | None = None,
        labels: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> BartForGenerationOutput:
        """
        Note that in Bart, if `decoder_input_ids` is not provided,
        it will be inferred from `labels` if `labels` is not `None`.
        """
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = model_utils.shift_tokens_right(labels, self.config.target_start_token_id)

            # replace [EOS] token with [PAD] token
            decoder_input_ids[decoder_input_ids == self.config.target_end_token_id] = self.config.target_pad_token_id

        outputs = self.model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_output=encoder_output,
        )  # (batch_size, seq_length, hidden_size)

        lm_logits = self.lm_head(outputs.decoder_output)
        lm_loss = None
        if labels is not None:
            lm_loss = self.get_lm_loss(lm_logits, labels=labels, label_smoothing=label_smoothing)

        return BartForGenerationOutput(
            encoder_output=outputs.encoder_output,
            decoder_output=outputs.decoder_output,
            lm_logits=lm_logits,
            lm_loss=lm_loss,
        )

    def get_lm_loss(
        self,
        lm_logits: Tensor,
        labels: Tensor | None = None,
        label_smoothing: float = 0.0
    ) -> Tensor | None:
        if labels is None:
            return None
        lm_loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.config.target_pad_token_id,
            label_smoothing=label_smoothing,
        )
        return lm_loss

class BartForNMT(BartBase):
    """Bart fine-tuned for Neural Machine Translation tasks."""
    def __init__(self, config: BartForNMTConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.model = Bart(config)

        # replace input embedding layer with new randomly initialized encoder
        del self.model.encoder.token_embeddings
        self.foreign_encoder = BertEncoderForForeignLanguage(config)

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
        assert self.lm_head.weight.shape == self.model.decoder.token_embeddings.weight.shape
        self.lm_head.weight = self.model.decoder.token_embeddings.weight

    def _init_model_weights(self, std: float = 0.02) -> None:
        self.apply(lambda module: initialize_bert_params_fn(module, std=std))

    def forward(
        self,
        encoder_input_ids: Tensor | None = None,
        encoder_attn_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attn_mask: Tensor | None = None,
        encoder_output: Tensor | None = None,
        labels: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> BartForGenerationOutput:
        """
        Note that in Bart, if `decoder_input_ids` is not provided,
        it will be inferred from `labels` if `labels` is not `None`.
        """
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = model_utils.shift_tokens_right(labels, self.config.target_start_token_id)

            # replace [EOS] token with [PAD] token
            decoder_input_ids[decoder_input_ids == self.config.target_end_token_id] = self.config.target_pad_token_id

        if encoder_output is None:
            if encoder_input_ids is None:
                raise ValueError('If `encoder_output` is not passed, `encoder_input_ids` can not be `None`')
            if encoder_attn_mask is None:
                encoder_attn_mask = encoder_input_ids != self.config.src_pad_token_id
            if encoder_attn_mask.dim() == 2:
                encoder_attn_mask = model_utils.create_encoder_4d_attn_mask(encoder_input_ids, encoder_attn_mask)
            foreign_encoder_output = self.foreign_encoder(
                encoder_input_ids,
                attn_mask=encoder_attn_mask,
            )
            encoder_output = self.model.encoder(
                foreign_encoder_output,
                attn_mask=encoder_attn_mask,
                embed_tokens=False,
            )

        assert encoder_output is not None
        decoder_output = None
        if decoder_input_ids is not None:
            if decoder_attn_mask is None:
                decoder_attn_mask = decoder_input_ids != self.config.target_pad_token_id
            if decoder_attn_mask.dim() == 2:
                decoder_attn_mask = model_utils.create_decoder_4d_attn_causal_mask(decoder_input_ids, decoder_attn_mask)
            decoder_output = self.model.decoder(
                decoder_input_ids,
                encoder_output=encoder_output,
                attn_mask=decoder_attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )

        lm_logits = self.lm_head(decoder_output)
        lm_loss = None
        if labels is not None:
            lm_loss = self.get_lm_loss(lm_logits, labels=labels, label_smoothing=label_smoothing)

        return BartForGenerationOutput(
            encoder_output=encoder_output,
            decoder_output=decoder_output,
            lm_logits=lm_logits,
            lm_loss=lm_loss,
        )

    def get_lm_loss(
        self,
        lm_logits: Tensor,
        labels: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> Tensor | None:
        if labels is None:
            return None
        lm_loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.config.target_pad_token_id,
            label_smoothing=label_smoothing,
        )
        return lm_loss

    def freeze_params(self) -> None:
        """
        Freeze all BART parameters, excluding: randomly initialized encoder,
        position embeddings, and the self-attention input projection matrix
        of BART's encoder first layer.
        """
        self.freezed_params = model_utils.freeze_parameters(self, exclude=[
            'foreign_encoder',
            'model.encoder.position_embedding',
            'model.encoder.layers.0.self_attention.w_q',
            'model.encoder.layers.0.self_attention.w_k',
            'model.encoder.layers.0.self_attention.w_v',
            'model.encoder.layers.0.self_attention.w_o',
        ])

    def unfreeze_params(self) -> None:
        if hasattr(self, 'freezed_params'):
            model_utils.unfreeze_parameters(self, self.freezed_params)
