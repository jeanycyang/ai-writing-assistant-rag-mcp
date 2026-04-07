import httpx

from services.agent_api.app import main


class HealthyRagClient:
    def healthcheck(self):
        return {"status": "ok", "base_url": "http://rag-api:8001"}


class HealthyProvider:
    def healthcheck(self):
        return {
            "status": "ok",
            "base_url": "http://host.docker.internal:11434",
            "model_configured": "hauhau-gemma4-e4b-q4km",
            "model_available": True,
        }


def test_healthz_reports_dependency_status(monkeypatch):
    monkeypatch.setattr(main, "RagApiClient", HealthyRagClient)
    monkeypatch.setattr(main, "get_llm_provider", lambda: HealthyProvider())

    payload = main.healthz()
    assert payload["status"] == "ok"
    assert payload["dependencies"]["rag_api"]["status"] == "ok"
    assert payload["dependencies"]["llm_provider"]["model_available"] is True


def test_healthz_degrades_when_dependency_fails(monkeypatch):
    class FailingRagClient:
        def healthcheck(self):
            raise httpx.ConnectError("rag down")

    class FailingProvider:
        def healthcheck(self):
            raise httpx.ConnectError("ollama down")

    monkeypatch.setattr(main, "RagApiClient", FailingRagClient)
    monkeypatch.setattr(main, "get_llm_provider", lambda: FailingProvider())

    payload = main.healthz()
    assert payload["status"] == "degraded"
    assert payload["dependencies"]["rag_api"]["status"] == "error"
    assert payload["dependencies"]["llm_provider"]["status"] == "error"
