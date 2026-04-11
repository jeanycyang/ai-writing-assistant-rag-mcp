from shared.config import Settings


def test_database_url_for_work_reuses_base_connection_and_swaps_database_name() -> None:
    settings = Settings(local_database_url="postgresql+psycopg://user:pass@localhost:5432/base_db")

    assert settings.database_url_for_work() == "postgresql+psycopg://user:pass@localhost:5432/base_db"
    assert settings.database_url_for_work("work_id_1") == "postgresql+psycopg://user:pass@localhost:5432/work_id_1"
