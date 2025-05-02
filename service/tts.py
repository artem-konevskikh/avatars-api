from typing import List

import torch
import torchaudio
from csm.generator import Segment, generate_streaming_audio, load_csm_1b

from src.models.avatars import Avatar


class CSM(object):
    def __init__(self) -> None:
        self._generator = load_csm_1b('cuda')
        self.avatar: Avatar | None = None
        self.segments: List[Segment] = []

    def load_avatar(self, avatar: Avatar) -> None:
        assert avatar.voice is not None
        self.avatar = avatar
        self.segments = [
            Segment(
                text='I knew I could trust you.',
                speaker=0,
                audio=self._load_audio(avatar.voice),
            ),
        ]

    def run(self, text: str, avatar: Avatar) -> None:
        self.load_avatar(avatar)
        generate_streaming_audio(
            generator=self._generator,
            text=text,
            speaker=0,
            context=self.segments,
            output_file='contextual_streaming.wav',
            play_audio=True,
        )

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0),
            orig_freq=sample_rate,
            new_freq=self._generator.sample_rate,
        )
        return audio_tensor
