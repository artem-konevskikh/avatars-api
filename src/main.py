import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.containers import AppContainer

def create_app() -> FastAPI:
    cfg = OmegaConf.load("config/config.yml")  # Omegaconf.DictConfig
    cfg = OmegaConf.to_container(cfg, resolve=True)  # dict
    container = AppContainer()
    container.config.from_dict(cfg)
    # container.wire([genre_routes])
    app = FastAPI()
    # set_routers(app)
    return app


def main() -> None:
    app = create_app()
    uvicorn.run(app, port=5004)


if __name__ == "__main__":
    main()
