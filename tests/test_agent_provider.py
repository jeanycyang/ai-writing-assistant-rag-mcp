from services.agent_api.app.provider import OllamaProvider
from shared.config import Settings


def test_ollama_complete_sets_keep_alive_to_zero() -> None:
    settings = Settings(
        ollama_base_url="http://example.test",
        ollama_model="demo-model",
        ollama_keep_alive="0s",
    )
    provider = OllamaProvider(settings)

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"role": "assistant", "content": "ok"}}

    class FakeClient:
        def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    provider._client = FakeClient()
    provider.complete([{"role": "user", "content": "test"}])

    assert captured["url"] == "http://example.test/api/chat"
    assert captured["json"]["keep_alive"] == "0s"
    assert captured["json"]["model"] == "demo-model"


def test_ollama_complete_omits_tools_when_not_requested() -> None:
    settings = Settings(
        ollama_base_url="http://example.test",
        ollama_model="demo-model",
        ollama_keep_alive="0s",
    )
    provider = OllamaProvider(settings)

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"role": "assistant", "content": "ok"}}

    class FakeClient:
        def post(self, url, json):
            captured["json"] = json
            return FakeResponse()

    provider._client = FakeClient()
    provider.complete([{"role": "user", "content": "test"}], tools=None, think=False)

    assert "tools" not in captured["json"]
    assert captured["json"]["think"] is False
    assert captured["json"]["options"]["num_predict"] == settings.ollama_num_predict
