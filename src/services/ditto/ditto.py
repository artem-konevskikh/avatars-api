class Ditto(object):
    def __init__(self, config: dict[str, str]) -> None:
        self._name = "Ditto"

    def get_name(self) -> str:
        return self._name

    def run(self) -> str:
        return f"{self._name} not implemented yet"
