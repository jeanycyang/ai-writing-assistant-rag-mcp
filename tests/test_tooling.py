from services.agent_api.app.provider import OllamaProvider
from shared.config import Settings
from shared.tooling import TOOL_SPECS


def test_tool_specs_are_adapted_for_ollama() -> None:
    provider = OllamaProvider(Settings())
    definitions = provider.build_tool_definitions()
    assert [item["function"]["name"] for item in definitions] == [spec.name for spec in TOOL_SPECS]
    assert definitions[0]["function"]["parameters"]["type"] == "object"
