import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf

from src.containers.containers import AppContainer
from src.api.routes import router as routes_router
from src.api.avatars import router as avatars_router
from src.api.tasks import router as tasks_router


def create_app(cfg: dict) -> FastAPI:
    container = AppContainer()
    container.config.from_dict(cfg)
    container.wire(modules=["src.api.routes", "src.api.avatars", "src.api.tasks"])
    app = FastAPI(title=cfg["app"]["title"], version=cfg["app"]["version"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes_router)
    app.include_router(avatars_router)
    app.include_router(tasks_router)

    return app


def main() -> None:
    cfg = OmegaConf.load("config/config.yml")  # Omegaconf.DictConfig
    cfg = OmegaConf.to_container(cfg, resolve=True)  # dict
    app = create_app(cfg)
    uvicorn.run(app, host=cfg["app"]["host"], port=cfg["app"]["port"])


if __name__ == "__main__":
    main()
