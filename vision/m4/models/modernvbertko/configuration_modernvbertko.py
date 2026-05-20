"""Configuration objects for ModernVBERT-Ko.

The defaults mirror the public ModernVBERT composition while swapping the text
backbone to SKT's Korean A.X encoder. As in the original modeling code, this
config stores numeric vocabulary/image-token metadata; tokenizer special-token
strings are expected to live in tokenizer artifacts.
"""

import copy
import os
from typing import Any, Dict, Mapping, Union

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

DEFAULT_TEXT_MODEL_NAME = "skt/A.X-Encoder-base"
DEFAULT_VISION_MODEL_NAME = "google/siglip2-base-patch16-512"


def _as_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if isinstance(config, PretrainedConfig):
        return config.to_dict()
    return {}


def collect_arg_in_candidates(config: Any, candidates: list[str], default: Any = None) -> Any:
    """Return the first matching attribute/key from a config-like object."""

    mapping = _as_mapping(config)
    for candidate in candidates:
        if hasattr(config, candidate):
            return getattr(config, candidate)
        if candidate in mapping:
            return mapping[candidate]
    if default is not None:
        return default
    raise ValueError(f"No matching arguments found in candidates={candidates}, config={config}")


class ModernVBertKoTextConfig(PretrainedConfig):
    """Text-backbone metadata for ModernVBERT-Ko."""

    model_type = "modernvbertko_text"

    def __init__(
        self,
        text_model_name: Union[str, os.PathLike] = DEFAULT_TEXT_MODEL_NAME,
        hidden_size: int = 768,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 12,
        intermediate_size: int = 1152,
        mlp_bias: bool = False,
        vocab_size: int = 50_000,
        max_position_embeddings: int = 16_384,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            text_model_name=str(text_model_name),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            mlp_bias=mlp_bias,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )

    @classmethod
    def from_base_model(
        cls,
        text_model_name: Union[str, os.PathLike] = DEFAULT_TEXT_MODEL_NAME,
        **kwargs: Any,
    ) -> "ModernVBertKoTextConfig":
        text_config = AutoConfig.from_pretrained(text_model_name, trust_remote_code=True)
        if hasattr(text_config, "text_config"):
            text_config = text_config.text_config

        return cls(
            text_model_name=text_model_name,
            hidden_size=collect_arg_in_candidates(text_config, ["hidden_size", "embed_dim"]),
            num_hidden_layers=collect_arg_in_candidates(
                text_config, ["num_hidden_layers", "num_hidden_blocks"]
            ),
            num_attention_heads=collect_arg_in_candidates(
                text_config, ["num_attention_heads", "num_heads"], default=12
            ),
            intermediate_size=collect_arg_in_candidates(
                text_config, ["intermediate_size", "mlp_dim"]
            ),
            mlp_bias=collect_arg_in_candidates(
                text_config, ["mlp_bias", "mlp_hidden_bias"], default=False
            ),
            vocab_size=collect_arg_in_candidates(text_config, ["vocab_size"]),
            max_position_embeddings=collect_arg_in_candidates(
                text_config, ["max_position_embeddings"], default=16_384
            ),
            **kwargs,
        )


class ModernVBertKoVisionConfig(PretrainedConfig):
    """Vision-backbone metadata for ModernVBERT-Ko."""

    model_type = "modernvbertko_vision"
    attribute_map = {"hidden_size": "embed_dim"}

    def __init__(
        self,
        vision_model_name: Union[str, os.PathLike] = DEFAULT_VISION_MODEL_NAME,
        embed_dim: int = 768,
        image_size: int = 512,
        patch_size: int = 16,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vision_model_name=str(vision_model_name),
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            **kwargs,
        )

    @classmethod
    def from_base_model(
        cls,
        vision_model_name: Union[str, os.PathLike] = DEFAULT_VISION_MODEL_NAME,
        **kwargs: Any,
    ) -> "ModernVBertKoVisionConfig":
        vision_config = AutoConfig.from_pretrained(vision_model_name, trust_remote_code=True)
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config

        return cls(
            vision_model_name=vision_model_name,
            embed_dim=collect_arg_in_candidates(vision_config, ["embed_dim", "hidden_size"]),
            image_size=collect_arg_in_candidates(vision_config, ["image_size", "img_size"]),
            patch_size=collect_arg_in_candidates(vision_config, ["patch_size"]),
            num_hidden_layers=collect_arg_in_candidates(
                vision_config, ["num_hidden_layers", "num_hidden_blocks"]
            ),
            num_attention_heads=collect_arg_in_candidates(
                vision_config, ["num_attention_heads", "num_heads"], default=12
            ),
            intermediate_size=collect_arg_in_candidates(
                vision_config, ["intermediate_size", "mlp_dim"]
            ),
            **kwargs,
        )


class ModernVBertKoConfig(PretrainedConfig):
    """Composite config for the Korean ModernVBERT adaptation."""

    model_type = "modernvbertko"
    is_composition = True

    def __init__(
        self,
        text_config: Union[PretrainedConfig, Dict[str, Any], None] = None,
        vision_config: Union[PretrainedConfig, Dict[str, Any], None] = None,
        image_token_id: int | None = None,
        vocab_size: int | None = None,
        base_vocab_size: int | None = None,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        freeze_config: Dict[str, bool] | None = None,
        pad_token_id: int | None = None,
        initializer_range: float = 0.02,
        pixel_shuffle_factor: int = 4,
        use_resampler: bool = False,
        additional_vocab_size: int = 40,
        neftune_noise_alpha: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.scale_factor = pixel_shuffle_factor
        self.additional_vocab_size = additional_vocab_size
        self.freeze_config = freeze_config or {
            "freeze_text_layers": False,
            "freeze_vision_layers": False,
        }
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.use_resampler = use_resampler
        self.neftune_noise_alpha = neftune_noise_alpha
        self.initializer_range = initializer_range

        if text_config is None:
            text_config = ModernVBertKoTextConfig()
        elif isinstance(text_config, dict):
            text_config = ModernVBertKoTextConfig.from_dict(text_config)
        self.text_config = text_config

        if vision_config is None:
            vision_config = ModernVBertKoVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = ModernVBertKoVisionConfig.from_dict(vision_config)
        self.vision_config = vision_config

        self.base_vocab_size = (
            self.text_config.vocab_size if base_vocab_size is None else base_vocab_size
        )
        if vocab_size is None:
            vocab_size = self.base_vocab_size + self.additional_vocab_size
        if image_token_id is None:
            image_token_id = vocab_size - 1
        self.image_token_id = image_token_id

        hidden_size = kwargs.pop("hidden_size", self.text_config.hidden_size)

        super().__init__(
            **kwargs,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output

    @classmethod
    def from_pretrained_models(
        cls,
        text_model_name: Union[str, os.PathLike] = DEFAULT_TEXT_MODEL_NAME,
        vision_model_name: Union[str, os.PathLike] = DEFAULT_VISION_MODEL_NAME,
        **kwargs: Any,
    ) -> "ModernVBertKoConfig":
        text_model_config = ModernVBertKoTextConfig.from_base_model(text_model_name)
        vision_model_config = ModernVBertKoVisionConfig.from_base_model(vision_model_name)
        return cls(
            text_config=text_model_config,
            vision_config=vision_model_config,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "ModernVBertKoConfig":
        """Build a fresh ModernVBERT-Ko config from m4 training arguments.

        m4 calls ``config_class.from_pretrained(hparams.model_name, **model_config)``
        for newly initialized pretraining runs. The alignment configs use
        ``model_name: modernvbertko`` and carry the real backbone names in
        ``model_config.text_model_name`` and ``model_config.vision_model_name``.
        """

        del model_args
        kwargs.pop("new_model", None)
        kwargs.pop("revision", None)
        kwargs.pop("trust_remote_code", None)

        text_config = kwargs.pop("text_config", None) or {}
        vision_config = kwargs.pop("vision_config", None) or {}
        freeze_config = kwargs.pop("freeze_config", None)
        if freeze_config is None:
            freeze_config = {}
        for key in (
            "freeze_lm_head",
            "freeze_text_layers",
            "freeze_text_module_exceptions",
            "freeze_vision_layers",
            "freeze_vision_module_exceptions",
        ):
            if key in kwargs:
                freeze_config[key] = kwargs.pop(key)

        text_model_name = (
            kwargs.pop("text_model_name", None)
            or getattr(text_config, "text_model_name", None)
            or (text_config.get("text_model_name") if isinstance(text_config, dict) else None)
            or str(pretrained_model_name_or_path)
        )
        if text_model_name in {"modernvbertko", "modernvbert-ko", cls.model_type}:
            text_model_name = DEFAULT_TEXT_MODEL_NAME
        vision_model_name = (
            kwargs.pop("vision_model_name", None)
            or getattr(vision_config, "vision_model_name", None)
            or (vision_config.get("vision_model_name") if isinstance(vision_config, dict) else None)
            or DEFAULT_VISION_MODEL_NAME
        )

        return cls.from_pretrained_models(
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            freeze_config=freeze_config or None,
            **kwargs,
        )
