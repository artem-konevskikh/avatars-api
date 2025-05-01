"""
Utility functions and classes for the CSM module.
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional, List, Tuple
import queue
import threading
import time

import torch
import numpy as np
import wave


from src.logger import LOGGER


@dataclass
class Segment:
    """A segment of speech with text and audio data."""
    speaker: int
    text: str
    audio: Optional[torch.Tensor] = None
    sample_rate: int = 24000


class AudioFormat(Enum):
    """Supported audio output formats."""
    WAV = auto()
    PCM = auto()
    TENSOR = auto()


def stream_audio_to_wav(
    filename: str,
    sample_rate: int,
) -> Tuple[Callable[[torch.Tensor], None], Callable[[], None]]:
    """
    Initialize a WAV writer for streaming audio chunks.

    Args:
        filename: Output WAV file path
        sample_rate: Audio sample rate in Hz

    Returns:
        tuple: (write_chunk, close) functions for writing audio data and closing the file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(filename).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create a WAV file with the proper header
    wav_file = wave.open(filename, "wb")
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)

    def write_chunk(audio_chunk: torch.Tensor) -> None:
        """Convert tensor to PCM format and write to WAV file."""
        # Ensure it's on CPU and detached before converting to numpy
        audio_np = audio_chunk.detach().cpu().numpy()

        # Normalize if needed (assuming audio is in [-1, 1] range)
        if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
            audio_int = (audio_np * 32767).astype(np.int16)
        else:
            audio_int = audio_np.astype(np.int16)

        # Write to WAV file
        wav_file.writeframes(audio_int.tobytes())

    def close() -> None:
        """Close the WAV file."""
        wav_file.close()

    return write_chunk, close


class AudioStreamProcessor:
    """Helper class for processing streaming audio with background thread."""

    def __init__(self, filename: str, sample_rate: int) -> None:
        """Initialize audio stream processor.

        Args:
            filename: Output file path
            sample_rate: Audio sample rate in Hz
        """
        self.filename = filename
        self.sample_rate = sample_rate
        self.audio_chunks: List[torch.Tensor] = []
        self.lock = threading.Lock()
        self.queue: queue.Queue = queue.Queue()
        self.running = True

        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def _writer_worker(self) -> None:
        """Background thread that handles audio chunk processing."""
        buffer_chunks: List[torch.Tensor] = []
        last_flush_time = time.time()

        while self.running or not self.queue.empty():
            try:
                # Get chunk with timeout to allow for regular checks
                chunk = self.queue.get(timeout=0.2)
                buffer_chunks.append(chunk)

                # Periodically flush the buffer to the main list
                current_time = time.time()
                if len(buffer_chunks) >= 10 or (
                    current_time - last_flush_time > 2.0 and buffer_chunks
                ):
                    with self.lock:
                        self.audio_chunks.extend(buffer_chunks)
                    buffer_chunks = []
                    last_flush_time = current_time

            except queue.Empty:
                # If queue is empty but we have pending chunks, add them
                if buffer_chunks:
                    with self.lock:
                        self.audio_chunks.extend(buffer_chunks)
                    buffer_chunks = []
                    last_flush_time = time.time()

        # Final flush of any remaining chunks
        if buffer_chunks:
            with self.lock:
                self.audio_chunks.extend(buffer_chunks)

    def add_chunk(self, chunk: torch.Tensor) -> None:
        """Add an audio chunk to the buffer queue without blocking."""
        try:
            self.queue.put(chunk, timeout=0.1)
        except queue.Full:
            # If queue is full, add directly to avoid losing data
            with self.lock:
                self.audio_chunks.append(chunk)

    def write_file(self) -> None:
        """Write all collected audio chunks to file and clean up."""
        # Signal the background thread to stop
        self.running = False
        # Wait for the thread to finish with a timeout
        self.writer_thread.join(timeout=3.0)

        with self.lock:
            if not self.audio_chunks:
                return

            # Concatenate all chunks
            audio = torch.cat(self.audio_chunks)

            # Create directories if needed
            output_dir = Path(self.filename).parent
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            # Save to file
            import torchaudio

            torchaudio.save(self.filename, audio.unsqueeze(0).cpu(), self.sample_rate)


def setup_optimizations() -> None:
    """Configure PyTorch optimizations for maximum performance."""
    # Enable TF32 on Ampere and later GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable flash attention if available
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        # Enable cudnn benchmarking and auto-tuner for best performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def get_optimal_dtype() -> torch.dtype:
    """Determine the optimal data type based on hardware capabilities."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def warmup_model(model_func: Callable, *args, **kwargs) -> None:
    """
    Perform model warmup by executing one small inference pass.

    Args:
        model_func: Function to execute for warmup
        *args, **kwargs: Arguments to pass to the function
    """
    LOGGER.info("Performing model warmup...")

    try:
        # Execute the function with provided arguments
        result = model_func(*args, **kwargs)

        # If the function returns a result, ensure it's properly released
        if isinstance(result, torch.Tensor):
            del result

        # Explicit garbage collection and CUDA cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        LOGGER.info("Model warmup completed successfully")
    except Exception as exc:
        LOGGER.error("Model warmup failed: {0}".format(exc))
