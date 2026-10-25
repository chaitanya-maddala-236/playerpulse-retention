import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PLAYERPULSE_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "playerpulse-synthetic-v1"


def _postgres_driver_url(value: str) -> str:
    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value.removeprefix("postgres://")
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value.removeprefix("postgresql://")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str
    model_dir: Path
    prediction_api_key: str
    player_id_hash_secret: str
    demo_mode: bool
    allowed_origins: tuple[str, ...]
    environment: str


def get_settings() -> Settings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
        if origin.strip()
    )
    return Settings(
        database_url=_postgres_driver_url(os.getenv("DATABASE_URL", "sqlite:///./playerpulse.db")),
        model_dir=Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR))),
        prediction_api_key=os.getenv("PREDICTION_API_KEY", ""),
        player_id_hash_secret=os.getenv(
            "PLAYER_ID_HASH_SECRET", "local-demo-hash-key-change-before-deploy"
        ),
        demo_mode=os.getenv("DEMO_MODE", "false").strip().lower() in {"1", "true", "yes"},
        allowed_origins=origins,
        environment=os.getenv("PLAYERPULSE_ENV", "development").strip().lower(),
    )
