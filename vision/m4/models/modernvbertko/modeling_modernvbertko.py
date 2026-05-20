"""PyTorch modeling code for ModernVBERT-Ko."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.utils import logging

from .configuration_modernvbertko import ModernVBertKoConfig


logger = logging.get_logger(__name__)


class DecoupledEmbedding(nn.Embedding):
    """Embedding layer with a frozen/base vocabulary plus trainable appended tokens."""

    def __init__(
        self,
        num_embeddings: int,
        num_additional_embeddings: int,
        embedding_dim: int,
        partially_freeze: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        padding_idx: int | None = None,
        **kwargs,
    ) -> None:
        if padding_idx is not None and padding_idx >= num_embeddings + num_additional_embeddings:
            raise ValueError(
                "padding_idx must be within the combined vocabulary. "
                f"Got padding_idx={padding_idx}, base={num_embeddings}, "
                f"additional={num_additional_embeddings}."
            )

        base_padding_idx = (
            padding_idx if padding_idx is not None and padding_idx < num_embeddings else None
        )
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=base_padding_idx,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze
        self.original_padding_idx = padding_idx

        if partially_freeze:
            self.weight.requires_grad_(False)

        if num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )

    @property
    def combined_num_embeddings(self) -> int:
        return self.num_embeddings + self.num_additional_embeddings

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if self.num_additional_embeddings == 0:
            return super().forward(input_ids)

        if torch.any(input_ids >= self.combined_num_embeddings):
            max_id = int(input_ids.max().item())
            raise IndexError(
                f"Input id {max_id} exceeds combined vocabulary size {self.combined_num_embeddings}."
            )

        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(
            input_ids_additional_vocab - self.num_embeddings
        )

        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)
        full_vector[additional_vocab_indices] = additional_embeddings
        return full_vector

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"num_additional_embeddings={self.num_additional_embeddings}, "
            f"embedding_dim={self.embedding_dim}, partially_freeze={self.partially_freeze}"
        )


@dataclass
class ModernVBertKoBaseModelOutput(BaseModelOutput):
    """ModernVBERT-Ko base-model output with optional image hidden states."""

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class ModernVBertKoMaskedLMOutput(MaskedLMOutput):
    """ModernVBERT-Ko masked-LM output with optional image hidden states."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class ModernVBertKoSimpleMLP(nn.Module):
    """Single linear projection from vision features into text hidden space."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ModernVBertKoConnector(nn.Module):
    """Pixel-shuffle connector used by the original ModernVBERT architecture."""

    def __init__(self, config: ModernVBertKoConfig) -> None:
        super().__init__()
        self.scale_factor = config.pixel_shuffle_factor
        self.modality_projection = ModernVBertKoSimpleMLP(
            input_size=config.vision_config.hidden_size * (self.scale_factor**2),
            output_size=config.text_config.hidden_size,
        )

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: int) -> torch.Tensor:
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        if height * width != seq:
            raise ValueError(
                f"Vision sequence length must be square before pixel shuffle. Got {seq}."
            )
        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError(
                f"Vision grid {height}x{width} is not divisible by scale_factor={scale_factor}."
            )

        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, width // scale_factor, embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            width // scale_factor,
            height // scale_factor,
            embed_dim * (scale_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        return x.reshape(bsz, seq // (scale_factor**2), embed_dim * (scale_factor**2))

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        return self.modality_projection(image_hidden_states)


class ModernVBertKoPreTrainedModel(PreTrainedModel):
    config_class = ModernVBertKoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ModernVBertKoModel(ModernVBertKoPreTrainedModel):
    def __init__(self, config: ModernVBertKoConfig) -> None:
        super().__init__(config)
        self.vision_model = self.init_vision_model(config)
        self.connector = ModernVBertKoConnector(config)
        self.text_model = self.init_language_model(config)
        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2)
            / (config.pixel_shuffle_factor**2)
        )
        self.image_token_id = config.image_token_id
        self._use_flash_attention_2 = (
            getattr(config, "_attn_implementation", None) == "flash_attention_2"
        )
        self._apply_freeze_config()
        self.vision_model.to(self.dtype)
        self.text_model.to(self.dtype)
        self.post_init()

    @staticmethod
    def _auto_config_kwargs(config: ModernVBertKoConfig) -> dict:
        attn_implementation = getattr(config, "_attn_implementation", None)
        return {"_attn_implementation": attn_implementation} if attn_implementation else {}

    @staticmethod
    def init_vision_model(config: ModernVBertKoConfig) -> nn.Module:
        vision_model_config = AutoConfig.from_pretrained(
            config.vision_config.vision_model_name,
            trust_remote_code=True,
            **ModernVBertKoModel._auto_config_kwargs(config),
        )
        vision_model = AutoModel.from_config(vision_model_config, trust_remote_code=True)
        return getattr(vision_model, "vision_model", vision_model)

    @staticmethod
    def init_language_model(config: ModernVBertKoConfig) -> nn.Module:
        text_model_config = AutoConfig.from_pretrained(
            config.text_config.text_model_name,
            trust_remote_code=True,
            **ModernVBertKoModel._auto_config_kwargs(config),
        )
        text_model = AutoModel.from_config(text_model_config, trust_remote_code=True)
        embed_layer = DecoupledEmbedding(
            num_embeddings=config.base_vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.text_config.hidden_size,
            partially_freeze=config.freeze_config.get("freeze_text_layers", False),
            padding_idx=config.pad_token_id,
        )
        text_model.set_input_embeddings(embed_layer)
        return text_model

    def _apply_freeze_config(self) -> None:
        if self.config.freeze_config.get("freeze_vision_layers", False):
            for parameter in self.vision_model.parameters():
                parameter.requires_grad_(False)
        if self.config.freeze_config.get("freeze_text_layers", False):
            for parameter in self.text_model.parameters():
                parameter.requires_grad_(False)
            embeddings = self.text_model.get_input_embeddings()
            if (
                isinstance(embeddings, DecoupledEmbedding)
                and embeddings.num_additional_embeddings > 0
            ):
                embeddings.additional_embedding.weight.requires_grad_(True)

    def enable_input_require_grads(self) -> None:
        def get_lowest_module(module: nn.Module) -> nn.Module:
            children = list(module.children())
            return module if not children else get_lowest_module(children[0])

        def make_inputs_require_grads(_module, _input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )
        self._vision_require_grads_hook = get_lowest_module(
            self.vision_model
        ).register_forward_hook(make_inputs_require_grads)

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.text_model.set_input_embeddings(value)

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Replace repeated ``<image>`` token embeddings with projected vision features."""

        _, patch_size, _ = image_hidden_states.shape
        image_mask = input_ids == self.image_token_id
        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("Number of <image> tokens not divisible by patch_size.")

        total_image_blocks = int((num_image_tokens // patch_size).sum().item())
        if total_image_blocks != image_hidden_states.shape[0]:
            raise ValueError(
                "Number of image-token blocks does not match image hidden states. "
                f"Got {total_image_blocks} token blocks and {image_hidden_states.shape[0]} image states."
            )

        blocks_per_sample = num_image_tokens // patch_size
        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[
            block_idx[image_mask], local_idx[image_mask], :
        ]
        return torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        del pixel_attention_mask
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        if pixel_values is not None:
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(1)
            batch_size, num_images, _, _, _ = pixel_values.shape
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            if not torch.any(real_images_inds):
                real_images_inds[0] = True
            pixel_values = pixel_values[real_images_inds].contiguous()
            image_hidden_states = self.vision_model(pixel_values=pixel_values).last_hidden_state
            image_hidden_states = self.connector(image_hidden_states)
        elif image_hidden_states is not None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=device)

        if inputs_embeds is not None and image_hidden_states is not None:
            if input_ids is None:
                raise ValueError("input_ids are required when merging image_hidden_states.")
            inputs_embeds = self.inputs_merger(input_ids, inputs_embeds, image_hidden_states)

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)
        return ModernVBertKoBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class ModernVBertKoLMHead(nn.Module):
    """Masked-LM head copied from the configured text backbone."""

    def __init__(self, config: ModernVBertKoConfig) -> None:
        super().__init__()
        pretrained_config = AutoConfig.from_pretrained(
            config.text_config.text_model_name,
            trust_remote_code=True,
        )
        pretrained_model = AutoModelForMaskedLM.from_config(
            pretrained_config, trust_remote_code=True
        )
        if hasattr(pretrained_model, "head") and hasattr(pretrained_model, "decoder"):
            self.head = pretrained_model.head
            self.decoder = pretrained_model.decoder
        elif hasattr(pretrained_model, "cls") and hasattr(pretrained_model.cls, "predictions"):
            self.head = pretrained_model.cls.predictions.transform
            self.decoder = pretrained_model.cls.predictions.decoder
        else:
            raise AttributeError(
                "Could not locate a ModernBERT-style MLM head/decoder on the text backbone."
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(hidden_states))


class ModernVBertKoForMaskedLM(ModernVBertKoPreTrainedModel):
    def __init__(self, config: ModernVBertKoConfig) -> None:
        super().__init__(config)
        self.image_token_id = config.image_token_id
        self.in_features = config.text_config.hidden_size
        self.out_additional_features = config.additional_vocab_size
        self.base_vocab_size = config.base_vocab_size
        self.vocab_size = config.vocab_size
        self.model = ModernVBertKoModel(config)
        self.lm_head = ModernVBertKoLMHead(config)
        if self.out_additional_features > 0:
            self.additional_fc = nn.Linear(
                self.in_features, self.out_additional_features, bias=False
            )
        freeze_config = getattr(config, "freeze_config", {}) or {}
        if freeze_config.get("freeze_lm_head", False):
            for parameter in self.lm_head.parameters():
                parameter.requires_grad_(False)
            if hasattr(self, "additional_fc"):
                for parameter in self.additional_fc.parameters():
                    parameter.requires_grad_(False)
        self.lm_head.to(self.dtype)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, ModernVBertKoMaskedLMOutput]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if self.out_additional_features > 0:
            proj_states = self.lm_head.head(hidden_states)
            additional_features = self.additional_fc(proj_states)
            logits = torch.cat((logits, additional_features), -1)

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return ModernVBertKoMaskedLMOutput(
            loss=loss,
            logits=logits.float(),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    @classmethod
    def from_pretrained_models(
        cls,
        model_name_or_path: str | None = None,
        *args,
        config: ModernVBertKoConfig | None = None,
        **kwargs,
    ):
        """Build a new ModernVBERT-Ko model for m4's pretraining path."""

        del args
        kwargs.pop("torch_dtype", None)
        if config is None:
            config = cls.config_class.from_pretrained(model_name_or_path or "modernvbertko")
        return cls(config)

