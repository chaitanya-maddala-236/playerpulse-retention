from alembic import context

from playerpulse.config import get_settings
from playerpulse.database import make_engine
from playerpulse.db_models import PredictionRecord

config = context.config
target_metadata = PredictionRecord.metadata
engine = make_engine(get_settings().database_url)


def run_migrations_offline() -> None:
    context.configure(
        url=engine.url, target_metadata=target_metadata, literal_binds=True, compare_type=True
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        with engine.connect() as connection:
            context.configure(
                connection=connection, target_metadata=target_metadata, compare_type=True
            )
            with context.begin_transaction():
                context.run_migrations()
    else:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
    engine.dispose()
