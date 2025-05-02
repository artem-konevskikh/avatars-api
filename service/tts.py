from datetime import datetime
from typing import List

import torch
import torchaudio

from service.csm.generator import Segment, generate_streaming_audio, load_csm_1b
from src.models.avatars import Avatar


class CSM(object):
    def __init__(self) -> None:
        self._generator = load_csm_1b('cuda')
        self.segments: List[Segment] = []

    def load_voice(self, voice_text: str, voice_path: str) -> None:
        self.segments = [
            Segment(
                text=voice_text,
                speaker=0,
                audio=self._load_audio(voice_path),
            ),
        ]

    def run(self, text: str, output_file: str) -> None:
        if len(self.segments) == 0:
            raise ValueError('No voice loaded')

        generate_streaming_audio(
            generator=self._generator,
            text=text,
            speaker=0,
            context=self.segments,
            output_file=output_file,
            play_audio=False,
        )

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0),
            orig_freq=sample_rate,
            new_freq=self._generator.sample_rate,
        )
        return audio_tensor


if __name__ == '__main__':
    avatar = Avatar(
        id='1',
        name='artem',
        bio='I am a test avatar.',
        voice='/home/aicu/ai/csm-streaming/voice.wav',
        created_at=datetime.now(),
    )
    csm = CSM()

    voice_text = 'The Manchurian hare (Lepus mandshuricus) is a species of mammal '
    'in the family Leporidae found in northeastern China and Russia, '
    'the Amur River basin, and possibly the mountains of northern North Korea.'
    voice_path = '/home/aicu/ai/csm-streaming/voice.wav'

    csm.load_voice(voice_text, voice_path)
    csm.run(
        "Hello! My name is Artem and I'm your digital avata! Ha-ha-ha!",
        'output.wav',
    )
