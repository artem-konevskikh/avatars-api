"""
Core CSM (Collaborative Speech Model) implementation.

This module provides the primary interface for the CSM text-to-speech system.
"""

import os
import time
from functools import lru_cache
from typing import List, Optional, Iterator, Callable, Tuple, cast

import numpy as np
import torch
import torchaudio

from src.logger import LOGGER
from src.services.csm.generator import CSMGenerator
from src.services.csm.models import Model, ModelArgs, ModelType
from src.services.csm.utils import (
    Segment,
    AudioFormat,
    stream_audio_to_wav,
    setup_optimizations,
    get_optimal_dtype,
    warmup_model,
)


class CSM(object):
    """
    Collaborative Speech Model (CSM) for high-quality text-to-speech synthesis.

    This class provides a simple interface for generating speech from text
    using the CSM model, with options for both streaming and non-streaming
    generation.
    """

    def __init__(self, config: dict[str, str]) -> None:
        """
        Initialize the CSM model.

        Args:
            config: Configuration object (optional, will use defaults if not provided)
        """
        self.config = config

        # Setup GPU optimizations
        setup_optimizations()

        # Initialize model and generator
        self._model: Optional[Model] = None
        self._generator: Optional[CSMGenerator] = None

        # Track model state
        self._is_loaded = False

        LOGGER.info("CSM initialized with config: {0}".format(self.config))

    def load_model(self) -> None:
        """
        Load the CSM model based on the configuration.

        This method will load either a pre-trained model from HuggingFace Hub
        or a local model from the specified path.
        """
        if self._is_loaded:
            LOGGER.info("Model already loaded, skipping")
            return

        start_time = time.time()
        LOGGER.info(
            "Loading {0} model on {1}".format(
                self.config["model_type"], self.config["device"]
            )
        )

        try:
            # Create model configuration
            model_config = ModelArgs(
                backbone_flavor="llama-1B",
                decoder_flavor="llama-100M",
                text_vocab_size=128256,
                audio_vocab_size=2051,
                audio_num_codebooks=self.config["audio_num_codebooks"],
            )

            # Determine if we're loading from local path
            is_local = (
                self.config["model_type"] == ModelType.CSM_LOCAL
                and self.config["local_model_path"]
            )

            # Load the model
            self._load_model(model_config, is_local)

            # Apply performance optimizations
            self._apply_optimizations()

            # Warm up the model
            self._warmup()

            LOGGER.info(
                "Model loaded successfully in {0:.2f}s".format(time.time() - start_time)
            )
            self._is_loaded = True

        except Exception as exc:
            LOGGER.error("Failed to load model: {0}".format(exc))
            raise

    def _load_model(self, model_config: ModelArgs, is_local: bool) -> None:
        """
        Load the CSM model from either HuggingFace Hub or a local path.

        Args:
            model_config: Model configuration
            is_local: Whether to load from a local path
        """

        # Determine model source
        model_path = self.config["local_model_path"] if is_local else "sesame/csm-1b"

        # Log loading information
        if is_local:
            LOGGER.info(f"Loading CSM model from local path: {model_path}")
        else:
            LOGGER.info("Loading CSM-1B model from HuggingFace Hub")

        # Create model instance
        model = Model(model_config)

        # Load model weights
        if is_local and self.config["local_model_path"]:
            model.from_pretrained(model_path)
        else:
            # If local_model_path is set, use it to cache the model
            if self.config["local_model_path"]:
                LOGGER.info(
                    f"Will save model to local path: {self.config['local_model_path']}"
                )
                # Create directory if it doesn't exist
                os.makedirs(
                    os.path.dirname(self.config["local_model_path"]), exist_ok=True
                )
                # Download and cache the model
                model.from_pretrained(
                    model_path, cache_dir=self.config["local_model_path"]
                )
            else:
                model.from_pretrained(model_path)

        # Move to correct device and dtype
        dtype = get_optimal_dtype()
        model.to(device=self.config["device"], dtype=dtype)
        model.eval()

        # Store model and create generator
        self._model = model
        self._generator = CSMGenerator(model)

        # Configure generator
        if self.config["optimize_for_streaming"]:
            self._generator._stream_buffer_size = self.config["stream_buffer_size"]

    def _apply_optimizations(self) -> None:
        """Apply performance optimizations to the model."""
        if not self._model or not self._generator:
            return

        # Apply torch.compile if enabled and available
        if self.config["compile_model"] and torch.cuda.is_available():
            try:
                LOGGER.info("Applying model compilation optimizations")
                # Compile decoder for better performance
                self._model.decoder = torch.compile(
                    self._model.decoder, fullgraph=True, backend="cudagraphs"
                )

                # Compile forward method
                self._model.forward = torch.compile(
                    self._model.forward, mode="max-autotune"
                )
                LOGGER.info("Model compilation complete")
            except Exception as exc:
                LOGGER.warning(
                    "Could not apply torch.compile optimizations: {0}".format(exc)
                )

        # Apply cache size configuration
        if self._generator:
            # Patch the tokenize method with configured cache size
            original_tokenize_text = self._generator._tokenize_text_segment

            @lru_cache(maxsize=self.config["cache_size"])
            def cached_tokenize_text_segment(
                text_str: str,
                speaker_int: int,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                return original_tokenize_text(text_str, speaker_int)

            self._generator._tokenize_text_segment = (
                lambda text, speaker: cached_tokenize_text_segment(text, speaker)
            )

    def _warmup(self) -> None:
        """
        Warm up the model with a small inference pass to initialize caches
        and optimize performance for subsequent calls.
        """
        if not self._generator:
            return

        # Perform a short generation to warm up the model
        warmup_text = "This is a quick test to warm up the model."

        def warmup_func() -> torch.Tensor:
            generator = cast(CSMGenerator, self._generator)  # Help type checker
            warmup_audio = next(
                generator.generate_stream(
                    text=warmup_text,
                    speaker=self.config["default_speaker_id"],
                    context=[],
                    max_audio_length_ms=5000,
                    temperature=self.config["temperature"],
                    topk=self.config["topk"],
                )
            )
            return warmup_audio

        warmup_model(warmup_func)

    def load_audio(self, audio_path: str) -> torch.Tensor:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0),
            orig_freq=sample_rate,
            new_freq=self._generator.sample_rate,
        )
        return audio_tensor

    def generate(
        self,
        text: str,
        output_file: Optional[str] = None,
        speaker_id: Optional[int] = None,
        context: Optional[List[Segment]] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        max_audio_length_ms: Optional[float] = None,
        output_format: AudioFormat = AudioFormat.TENSOR,
    ) -> Optional[torch.Tensor]:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            output_file: Path to save the generated audio (optional)
            speaker_id: Speaker ID (optional, uses default if not provided)
            context: List of context segments for continuation (optional)
            temperature: Sampling temperature (optional, uses config default if not provided)
            topk: Top-k sampling parameter (optional, uses config default if not provided)
            max_audio_length_ms: Maximum audio length in milliseconds (optional)
            output_format: Output format (TENSOR, WAV, PCM)

        Returns:
            Audio tensor if output_format is TENSOR, otherwise None
        """
        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()

        # Check if generator is available
        if not self._generator:
            raise RuntimeError("Generator not initialized")

        # Use default values if not provided
        speaker_id = (
            speaker_id if speaker_id is not None else self.config["default_speaker_id"]
        )
        temperature = (
            temperature if temperature is not None else self.config["temperature"]
        )
        topk = topk if topk is not None else self.config["topk"]
        max_audio_length_ms = (
            max_audio_length_ms
            if max_audio_length_ms is not None
            else self.config["max_audio_length_ms"]
        )
        context = context if context is not None else []

        # Generate audio
        LOGGER.info(
            "Generating audio for text: {0}".format(
                text[:50] + ("..." if len(text) > 50 else "")
            )
        )
        audio = self._generator.generate(
            text=text,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )

        # Save to file if requested
        if output_file and output_format != AudioFormat.TENSOR:
            self._save_audio(audio, output_file, output_format)
            return None

        return audio

    def generate_stream(
        self,
        text: str,
        output_file: Optional[str] = None,
        speaker_id: Optional[int] = None,
        context: Optional[List[Segment]] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        max_audio_length_ms: Optional[float] = None,
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Generate speech from text in a streaming fashion.

        Args:
            text: Text to synthesize
            output_file: Path to save the generated audio (optional)
            speaker_id: Speaker ID (optional, uses default if not provided)
            context: List of context segments for continuation (optional)
            temperature: Sampling temperature (optional, uses config default if not provided)
            topk: Top-k sampling parameter (optional, uses config default if not provided)
            max_audio_length_ms: Maximum audio length in milliseconds (optional)
            on_chunk_generated: Callback function for each generated chunk (optional)

        Yields:
            Audio chunks as they are generated
        """
        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()

        # Check if generator is available
        if not self._generator:
            raise RuntimeError("Generator not initialized")

        # Use default values if not provided
        speaker_id = (
            speaker_id if speaker_id is not None else self.config["default_speaker_id"]
        )
        temperature = (
            temperature if temperature is not None else self.config["temperature"]
        )
        topk = topk if topk is not None else self.config["topk"]
        max_audio_length_ms = (
            max_audio_length_ms
            if max_audio_length_ms is not None
            else self.config["max_audio_length_ms"]
        )
        context = context if context is not None else []

        # Setup streaming to file if requested
        file_writer = None
        close_writer = None

        if output_file:
            file_writer, close_writer = stream_audio_to_wav(
                output_file, self._generator.sample_rate
            )

            # Create a combined callback that writes to file and calls the user callback
            original_callback = on_chunk_generated

            def combined_callback(chunk: torch.Tensor) -> None:
                if file_writer:
                    file_writer(chunk)
                if original_callback:
                    original_callback(chunk)

            on_chunk_generated = combined_callback

        try:
            # Generate audio in streaming mode
            LOGGER.info(
                "Streaming audio for text: {0}".format(
                    text[:50] + ("..." if len(text) > 50 else "")
                )
            )
            for chunk in self._generator.generate_stream(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                topk=topk,
                on_chunk_generated=on_chunk_generated,
            ):
                yield chunk

        finally:
            # Close file writer if it was created
            if close_writer:
                close_writer()

    def _save_audio(
        self, audio: torch.Tensor, output_file: str, format_type: AudioFormat
    ) -> None:
        """
        Save audio tensor to file.

        Args:
            audio: Audio tensor
            output_file: Output file path
            format_type: Output format
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if format_type == AudioFormat.WAV:
            # Save as WAV file
            if not self._generator:
                raise RuntimeError("Generator not initialized")

            torchaudio.save(
                output_file, audio.unsqueeze(0).cpu(), self._generator.sample_rate
            )
            LOGGER.info("Saved audio to {0}".format(output_file))

        elif format_type == AudioFormat.PCM:
            # Save as raw PCM data
            audio_np = audio.detach().cpu().numpy()

            # Normalize to int16 range
            if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
                audio_int = (audio_np * 32767).astype(np.int16)
            else:
                audio_int = audio_np.astype(np.int16)

            # Write to file
            with open(output_file, "wb") as f:
                f.write(audio_int.tobytes())
            LOGGER.info("Saved PCM audio to {0}".format(output_file))
