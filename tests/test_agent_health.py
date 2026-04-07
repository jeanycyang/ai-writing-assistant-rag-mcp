import httpx
import pytest

from services.agent_api.app import main
from services.agent_api.app.client import RagApiClient


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


def test_healthz_reports_process_up():
    payload = main.healthz()
    assert payload["status"] == "ok"


def test_startup_preloads_embedding_provider(monkeypatch):
    called = {"value": False}

    def fake_preload():
        called["value"] = True

    monkeypatch.setattr(main, "preload_embedding_provider", fake_preload)

    main.startup()
    assert called["value"] is True


def test_readyz_reports_dependency_status(monkeypatch):
    monkeypatch.setattr(main, "RagApiClient", HealthyRagClient)
    monkeypatch.setattr(main, "get_llm_provider", lambda: HealthyProvider())

    payload = main.readyz()
    assert payload["status"] == "ok"
    assert payload["dependencies"]["rag_api"]["status"] == "ok"
    assert payload["dependencies"]["llm_provider"]["model_available"] is True


def test_readyz_degrades_when_dependency_fails(monkeypatch):
    class FailingRagClient:
        def healthcheck(self):
            raise httpx.ConnectError("rag down")

    class FailingProvider:
        def healthcheck(self):
            raise httpx.ConnectError("ollama down")

    monkeypatch.setattr(main, "RagApiClient", FailingRagClient)
    monkeypatch.setattr(main, "get_llm_provider", lambda: FailingProvider())

    payload = main.readyz()
    assert payload["status"] == "degraded"
    assert payload["dependencies"]["rag_api"]["status"] == "error"
    assert payload["dependencies"]["llm_provider"]["status"] == "error"


def test_readyz_degrades_when_model_is_unavailable(monkeypatch):
    class HealthyButMissingModelRagClient:
        def healthcheck(self):
            return {"status": "ok", "base_url": "http://rag-api:8001"}

    class HealthyButMissingModelProvider:
        def healthcheck(self):
            return {
                "status": "ok",
                "base_url": "http://host.docker.internal:11434",
                "model_configured": "hauhau-gemma4-e4b-q4km",
                "model_available": False,
            }

    monkeypatch.setattr(main, "RagApiClient", HealthyButMissingModelRagClient)
    monkeypatch.setattr(main, "get_llm_provider", lambda: HealthyButMissingModelProvider())

    payload = main.readyz()
    assert payload["status"] == "degraded"
    assert payload["dependencies"]["rag_api"]["status"] == "ok"
    assert payload["dependencies"]["llm_provider"]["model_available"] is False


def test_readyz_accepts_ollama_latest_tag(monkeypatch):
    class HealthyTaggedRagClient:
        def healthcheck(self):
            return {"status": "ok", "base_url": "http://rag-api:8001"}

    class HealthyTaggedProvider:
        def healthcheck(self):
            return {
                "status": "ok",
                "base_url": "http://host.docker.internal:11434",
                "model_configured": "hauhau-gemma4-e4b-q4km",
                "model_available": True,
            }

    monkeypatch.setattr(main, "RagApiClient", HealthyTaggedRagClient)
    monkeypatch.setattr(main, "get_llm_provider", lambda: HealthyTaggedProvider())

    payload = main.readyz()
    assert payload["status"] == "ok"


def test_rag_client_healthcheck_does_not_initialize_embeddings(monkeypatch):
    def fail_if_called():
        raise AssertionError("embedding provider should not be initialized during healthcheck")

    monkeypatch.setattr("services.agent_api.app.client.get_embedding_provider", fail_if_called)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok"}

    class FakeHttpClient:
        def get(self, path):
            assert path == "/healthz"
            return FakeResponse()

    client = RagApiClient()
    client._client = FakeHttpClient()

    payload = client.healthcheck()
    assert payload["status"] == "ok"
