from src.logger import logger


class CSM(object):
    def __init__(self, config: dict[str, str]) -> None:
        self._name = "CSM"

    def get_name(self) -> str:
        return self._name

    def run(self, text: str) -> str:
        logger.info(f"Running {self._name}: {text}")
        return "dummy_voice.wav"
