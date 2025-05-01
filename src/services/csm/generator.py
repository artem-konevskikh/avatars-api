"""
Generator module for CSM text-to-speech synthesis.
"""

from functools import lru_cache
import time
from typing import List, Tuple, Optional, Callable, Dict, Iterator

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from src.logger import LOGGER
from src.services.csm.models import Model
from src.services.csm.utils import Segment


def load_llama3_tokenizer() -> AutoTokenizer:
    """
    Load and configure the Llama 3 tokenizer.

    Returns:
        Configured AutoTokenizer instance
    """
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id),
            (f"{eos}", tokenizer.eos_token_id),
        ],
    )

    return tokenizer


class CSMGenerator:
    """
    Text-to-speech generator using the CSM model.

    This class handles tokenization, inference, and audio generation using
    the Collaborative Speech Model.
    """

    def __init__(self, model: Model) -> None:
        """
        Initialize the CSM generator.

        Args:
            model: Loaded CSM model instance
        """
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()
        self.device = next(model.parameters()).device

        # Load MIMI vocoder
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=self.device)

        num_codebooks = model.config.audio_num_codebooks
        mimi.set_num_codebooks(num_codebooks)
        self._num_codebooks = num_codebooks
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self._stream_buffer_size = 20
        self.max_seq_len = 2048
        self._text_token_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @lru_cache(maxsize=2048)
    def _tokenize_text_segment_cached(
        self,
        text: str,
        speaker: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize text segment with caching for reduced latency (cached version).

        Args:
            text: Text to tokenize
            speaker: Speaker ID

        Returns:
            Tuple of token tensors and mask tensors
        """
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(
            len(text_tokens),
            self._num_codebooks + 1,
            dtype=torch.long,
            device=self.device,
        )
        text_frame_mask = torch.zeros(
            len(text_tokens),
            self._num_codebooks + 1,
            dtype=torch.bool,
            device=self.device,
        )
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True

        return text_frame, text_frame_mask

    def _tokenize_text_segment(
        self,
        text: str,
        speaker: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize text segment with caching for reduced latency.

        Args:
            text: Text to tokenize
            speaker: Speaker ID

        Returns:
            Tuple of token tensors and mask tensors
        """
        return self._tokenize_text_segment_cached(text, speaker)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize audio into model tokens.

        Args:
            audio: Audio tensor

        Returns:
            Tuple of token tensors and mask tensors
        """
        # Move audio to the correct device
        audio = audio.to(self.device)

        # Encode the audio using the tokenizer (K, T)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]

        # Limit to the number of codebooks set in MIMI
        audio_tokens = audio_tokens[: self._num_codebooks, :]

        # Add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1, device=self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        # Create frame and mask
        audio_frame = torch.zeros(
            audio_tokens.size(1),
            self._num_codebooks + 1,
            dtype=torch.long,
            device=self.device,
        )
        audio_frame_mask = torch.zeros(
            audio_tokens.size(1),
            self._num_codebooks + 1,
            dtype=torch.bool,
            device=self.device,
        )

        # Fill in the frame and mask
        audio_frame[:, : self._num_codebooks] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, : self._num_codebooks] = True

        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a full segment (text and optional audio).

        Args:
            segment: Segment to tokenize

        Returns:
            Tuple of token tensors and mask tensors
        """
        text_tokens, text_masks = self._tokenize_text_segment(
            segment.text, segment.speaker
        )

        if segment.audio is not None:
            audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
            total_len = text_tokens.size(0) + audio_tokens.size(0)

            # Handle sequence length constraints
            if total_len > self.max_seq_len:
                overflow = total_len - self.max_seq_len

                if text_tokens.size(0) > overflow:
                    text_tokens = text_tokens[overflow:]
                    text_masks = text_masks[overflow:]
                else:
                    audio_overflow = overflow - text_tokens.size(0)
                    text_tokens = text_tokens[0:0]
                    text_masks = text_masks[0:0]
                    audio_tokens = audio_tokens[audio_overflow:]
                    audio_masks = audio_masks[audio_overflow:]

            return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat(
                [text_masks, audio_masks], dim=0
            )

        return text_tokens, text_masks

    def _decode_frames(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode audio frames to waveform.

        Args:
            frames: List of token frames

        Returns:
            Audio waveform tensor
        """
        if not frames:
            return torch.tensor([], device=self.device)

        # Only use first half of codebooks for faster decoding
        frames_reduced = [frame[:, : self._num_codebooks // 2] for frame in frames]
        audio = (
            self._audio_tokenizer.decode(torch.stack(frames_reduced).permute(1, 2, 0))
            .squeeze(0)
            .squeeze(0)
        )
        return audio

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Generate audio in a streaming fashion, yielding chunks as they're produced.

        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of context segments for continuation
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature (higher = more random)
            topk: Number of top logits to consider for sampling
            on_chunk_generated: Optional callback function for each chunk

        Yields:
            Audio chunks as they are generated
        """
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)

        tokens, tokens_mask = [], []

        # Configuration for real-time streaming
        initial_batch_size = 20
        normal_batch_size = 20
        initial_buffer_size = 20
        normal_buffer_size = 20

        batch_size = initial_batch_size
        buffer_size = initial_buffer_size
        first_chunk_delivered = False

        # Process context segments
        if context:
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

        # Add the new text segment to generate
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
            text, speaker
        )
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        # Combine all tokens
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        # Ensure we don't exceed max sequence length
        if prompt_tokens.size(0) > self.max_seq_len:
            prompt_tokens = prompt_tokens[-self.max_seq_len :]
            prompt_tokens_mask = prompt_tokens_mask[-self.max_seq_len :]

        # Initialize with prompt tokens
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = (
            torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        )

        # Setup for frame generation
        expected_frame_count = buffer_size
        frame_buffer = []

        # Constants for token updates
        zeros_1_1 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        zeros_mask_1_1 = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

        # Helper function to update tokens after generating a frame
        def update_tokens(sample: torch.Tensor) -> None:
            nonlocal curr_tokens, curr_tokens_mask, curr_pos
            ones = torch.ones_like(sample, dtype=torch.bool)
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        # Enable streaming mode for the audio tokenizer
        with self._audio_tokenizer.streaming(1):
            i = 0
            generation_start = time.time()

            while i < max_generation_len:
                batch_end = min(i + batch_size, max_generation_len)
                batch_size_actual = batch_end - i

                batch_samples = []

                # Generate batch_size_actual frames
                for _ in range(batch_size_actual):
                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.bfloat16
                    ):
                        sample = self._model.generate_frame(
                            curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                        )

                        # Check for NaN or empty values
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.synchronize()
                                if sample.numel() == 0 or torch.isnan(sample).any():
                                    LOGGER.warning(
                                        "Generated empty or NaN sample, stopping generation"
                                    )
                                    break
                            except Exception as exc:
                                LOGGER.error("Error checking tensor: {0}".format(exc))
                                break

                    # Check for EOS token (all zeros)
                    if torch.all(sample == 0):
                        break

                    batch_samples.append(sample)
                    update_tokens(sample)

                if not batch_samples:
                    break

                # Add samples to the buffer
                frame_buffer.extend(batch_samples)
                i += len(batch_samples)

                # When we have enough frames, process and yield them
                if len(frame_buffer) >= buffer_size:
                    frames_to_process = frame_buffer[:expected_frame_count]

                    # If we don't have enough frames, pad with zeros to match expected shape
                    if len(frames_to_process) < expected_frame_count:
                        padding_frames = [
                            torch.zeros_like(frames_to_process[0])
                            for _ in range(
                                expected_frame_count - len(frames_to_process)
                            )
                        ]
                        frames_to_process = frames_to_process + padding_frames

                    frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                    audio_chunk = (
                        self._audio_tokenizer.decode(frames_stacked)
                        .squeeze(0)
                        .squeeze(0)
                    )

                    # Keep remaining frames for next iteration
                    frame_buffer = frame_buffer[expected_frame_count:]

                    # Process and yield the chunk
                    cpu_chunk = audio_chunk.cpu()
                    if on_chunk_generated:
                        on_chunk_generated(cpu_chunk)

                    # After first chunk is delivered, switch to normal batch and buffer sizes
                    if not first_chunk_delivered:
                        batch_size = normal_batch_size
                        buffer_size = normal_buffer_size
                        expected_frame_count = buffer_size
                        first_chunk_delivered = True

                    yield cpu_chunk

                    # Occasionally log progress and sync GPU
                    if i >= 100 and (i % 100 == 0):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        LOGGER.info(
                            "Generated {0} frames ({1:.2f}s of audio)".format(
                                i, i * 0.08
                            )
                        )

            # Process any remaining frames
            if frame_buffer:
                # Pad frame buffer if necessary
                if len(frame_buffer) < expected_frame_count:
                    padding_frames = [
                        torch.zeros_like(frame_buffer[0])
                        for _ in range(expected_frame_count - len(frame_buffer))
                    ]
                    frames_to_process = frame_buffer + padding_frames
                else:
                    # Otherwise take as many frames as possible that are a multiple of expected_frame_count
                    frames_multiple = (
                        len(frame_buffer) // expected_frame_count
                    ) * expected_frame_count
                    frames_to_process = frame_buffer[:frames_multiple]

                frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                audio_chunk = (
                    self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                )

                # Determine actual audio length (before padding)
                actual_frames_percentage = (
                    min(len(frame_buffer), expected_frame_count) / expected_frame_count
                )
                actual_samples = int(audio_chunk.shape[0] * actual_frames_percentage)

                # Return only the non-padded portion of audio if we added padding
                if len(frame_buffer) < expected_frame_count:
                    audio_chunk = audio_chunk[:actual_samples]

                cpu_chunk = audio_chunk.cpu()
                if on_chunk_generated:
                    on_chunk_generated(cpu_chunk)
                yield cpu_chunk

            # Log final performance metrics
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.time() - generation_start
            frames_generated = i
            audio_seconds = frames_generated * 0.08
            rtf = total_time / audio_seconds if audio_seconds > 0 else float("inf")
            LOGGER.info("Total time: {0:.2f}s".format(total_time))
            LOGGER.info(
                "Generated {0} frames ({1:.2f}s of audio)".format(
                    frames_generated, audio_seconds
                )
            )
            LOGGER.info("Real-time factor: {0:.3f}x (target: <1.0)".format(rtf))

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: Optional[List[Segment]] = None,
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.8,
        topk: int = 40,
    ) -> torch.Tensor:
        """
        Generate audio in a non-streaming fashion.

        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of context segments for continuation
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature (higher = more random)
            topk: Number of top logits to consider for sampling

        Returns:
            Complete audio waveform tensor
        """
        if context is None:
            context = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        # Process context segments
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        # Add the new text segment to generate
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(
            text, speaker
        )
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        # Combine all tokens
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        # Ensure we don't exceed max sequence length
        if prompt_tokens.size(0) > self.max_seq_len:
            prompt_tokens = prompt_tokens[-self.max_seq_len :]
            prompt_tokens_mask = prompt_tokens_mask[-self.max_seq_len :]

        # Initialize with prompt tokens
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = (
            torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        )

        samples = []
        with self._audio_tokenizer.streaming(1):
            for i in range(max_generation_len):
                sample = self._model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                )
                if torch.all(sample == 0):
                    break
                samples.append(sample)

                curr_tokens = torch.cat(
                    [sample, torch.zeros(1, 1).long().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [
                        torch.ones_like(sample).bool(),
                        torch.zeros(1, 1).bool().to(self.device),
                    ],
                    dim=1,
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            return torch.tensor([], device=self.device)

        return (
            self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))
            .squeeze(0)
            .squeeze(0)
        )
