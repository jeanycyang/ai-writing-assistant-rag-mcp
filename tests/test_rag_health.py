from services.rag_api.app import main


class FakeSession:
    def __init__(self) -> None:
        self.executed = False

    def execute(self, statement) -> None:
        self.executed = True


def test_healthz_reports_process_up():
    payload = main.healthz()
    assert payload["status"] == "ok"


def test_readyz_checks_database():
    session = FakeSession()
    payload = main.readyz(session)
    assert payload["status"] == "ok"
    assert session.executed is True
