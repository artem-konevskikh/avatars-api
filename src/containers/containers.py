from dependency_injector import containers, providers

from src.db.database import Database
from src.services.csm.csm import CSM
from src.services.ditto.ditto import Ditto
from src.services.avatar_service import AvatarService
from src.services.task_service import TaskService


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    db = providers.Singleton(Database)

    tts = providers.Singleton(
        CSM,
        config=config.csm,
    )
    talking_head = providers.Singleton(
        Ditto,
        config=config.ditto,
    )
    avatar_service = providers.Singleton(
        AvatarService,
        db=db,
    )
    task_service = providers.Singleton(
        TaskService,
        db=db,
        tts=tts,
        talking_head=talking_head,
    )
