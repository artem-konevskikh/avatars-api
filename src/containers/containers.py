from dependency_injector import containers, providers

from src.services.csm.csm import CSM
from src.services.ditto.ditto import Ditto


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    text_to_voice = providers.Singleton(
        CSM,
        config=config.csm,
    )
    talking_head = providers.Singleton(
        Ditto,
        config=config.ditto,
    )
