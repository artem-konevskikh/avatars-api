from src.logger import logger


class Ditto(object):
    def __init__(self, config: dict[str, str]) -> None:
        self._name = "Ditto"

    def get_name(self) -> str:
        return self._name

    def run(self, audio_file: str) -> str:
        logger.info(f"Running {self._name}: {audio_file}")
        return "dummy_video.mp4"
