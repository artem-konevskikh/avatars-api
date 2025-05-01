"""
Model definitions and utilities for CSM.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torchtune.models import llama3_2
from torchtune.modules.transformer import TransformerDecoder


class ModelType(Enum):
    """Available model types for CSM."""
    CSM_1B = auto()
    CSM_LOCAL = auto()


def llama3_2_1b() -> TransformerDecoder:
    """Create a 1B parameter Llama 3.2 model configuration."""
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100m() -> TransformerDecoder:
    """Create a 100M parameter Llama 3.2 model configuration."""
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS: Dict[str, Callable[[], TransformerDecoder]] = {
    "llama-1B": llama3_2_1b,
    "llama-100M": llama3_2_100m,
}


def _prepare_transformer(model: nn.Module) -> Tuple[nn.Module, int]:
    """
    Prepare a transformer model for use in CSM.

    Args:
        model: The transformer model to prepare

    Returns:
        Tuple containing the prepared model and embedding dimension
    """
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for transformer attention.

    Args:
        seq_len: Sequence length
        device: Device to create the mask on

    Returns:
        Boolean causal mask tensor
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    """
    Index into a causal mask using position indices.

    Args:
        mask: Causal mask of shape (max_seq_len, max_seq_len)
        input_pos: Position indices of shape (batch_size, seq_len)

    Returns:
        Indexed mask of shape (batch_size, seq_len, max_seq_len)
    """
    return mask[input_pos, :]


def _multinomial_sample_one_no_sync(probs: torch.Tensor) -> torch.Tensor:
    """
    Perform multinomial sampling without forcing a CUDA synchronization.

    Args:
        probs: Probability distribution to sample from

    Returns:
        Sampled token indices
    """
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    """
    Sample from a distribution using top-k sampling with temperature.

    Args:
        logits: Raw logits
        topk: Number of top logits to consider
        temperature: Sampling temperature (higher = more random)

    Returns:
        Sampled token indices
    """
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    """Configuration for CSM model architecture."""

    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/SesameAILabs/csm",
    pipeline_tag="text-to-speech",
    license="apache-2.0",
):
    """
    Collaborative Speech Model (CSM) implementation.

    This model uses a transformer-based architecture to generate speech from text.
    """

    def __init__(self, config: ModelArgs) -> None:
        """Initialize the CSM model.

        Args:
            config: Configuration for model architecture
        """
        super().__init__()
        self.config = config

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )

    def setup_caches(self, max_batch_size: int) -> None:
        """
        Setup KV caches for efficient inference.

        Args:
            max_batch_size: Maximum batch size for inference
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(
                max_batch_size,
                dtype,
                decoder_max_seq_len=self.config.audio_num_codebooks,
            )

        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(self.backbone.max_seq_len, device),
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, device),
        )

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Generate a single frame of audio tokens.

        Args:
            tokens: Input tokens of shape (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: Mask for input tokens of shape (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: Position indices of shape (batch_size, seq_len)
            temperature: Sampling temperature
            topk: Number of top logits to consider for sampling

        Returns:
            Generated audio token frame of shape (batch_size, audio_num_codebooks)
        """
        dtype = next(self.parameters()).dtype
        _batch_size, _seq_len, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        hidden_state = masked_embeds.sum(dim=2)
        hidden_state = self.backbone(
            hidden_state, input_pos=input_pos, mask=curr_backbone_mask
        ).to(dtype=dtype)

        last_h = hidden_state[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(curr_h.size(0), 1)
        )

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            ).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self) -> None:
        """Reset KV caches for the backbone and decoder."""
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed audio tokens for a specific codebook.

        Args:
            codebook: Codebook index
            tokens: Token indices to embed

        Returns:
            Embedded tokens
        """
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed both text and audio tokens.

        Args:
            tokens: Combined token indices

        Returns:
            Embedded tokens
        """
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
