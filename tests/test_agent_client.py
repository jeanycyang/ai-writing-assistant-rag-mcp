from services.agent_api.app.client import RagApiClient


def test_normalize_linked_raw_payload_accepts_hit_objects() -> None:
    client = RagApiClient()

    payload = client._normalize_linked_raw_payload(
        {
            "summary_hit_ids": [
                {"id": "11111111-1111-1111-1111-111111111111"},
                {"summary_id": "22222222-2222-2222-2222-222222222222"},
            ],
            "top_k": 1,
        }
    )

    assert payload == {
        "summary_hit_ids": [
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
        ],
        "top_k_per_hit": 1,
    }


def test_normalize_linked_raw_payload_accepts_alternate_hit_keys() -> None:
    client = RagApiClient()

    payload = client._normalize_linked_raw_payload(
        {
            "hits": {"summary_hit_id": "33333333-3333-3333-3333-333333333333"},
            "top_k_per_hit": 2,
        }
    )

    assert payload == {
        "summary_hit_ids": ["33333333-3333-3333-3333-333333333333"],
        "top_k_per_hit": 2,
    }


def test_get_raw_paragraph_posts_exact_metadata_lookup() -> None:
    client = RagApiClient()
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"hits": []}

    class FakeHttpClient:
        def post(self, path, json):
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    client._client = FakeHttpClient()
    client.get_raw_paragraph({"chapter_id": "Chapter_16", "paragraph_id": 18})

    assert captured == {
        "path": "/retrieve/raw-paragraph",
        "json": {"chapter_id": "Chapter_16", "paragraph_id": 18},
    }


def test_get_summary_paragraph_posts_exact_metadata_lookup() -> None:
    client = RagApiClient()
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"hits": []}

    class FakeHttpClient:
        def post(self, path, json):
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    client._client = FakeHttpClient()
    client.get_summary_paragraph({"chapter_id": "Chapter_16", "paragraph_id": 18})

    assert captured == {
        "path": "/retrieve/summary-paragraph",
        "json": {"chapter_id": "Chapter_16", "paragraph_id": 18},
    }


def test_get_summary_chapter_posts_chapter_lookup() -> None:
    client = RagApiClient()
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"chapter_id": "Chapter_16", "paragraphs": [], "full_summary_text": ""}

    class FakeHttpClient:
        def post(self, path, json):
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    client._client = FakeHttpClient()
    client.get_summary_chapter({"chapter_id": "Chapter_16"})

    assert captured == {
        "path": "/retrieve/summary-chapter",
        "json": {"chapter_id": "Chapter_16"},
    }


def test_get_raw_chapter_posts_chapter_lookup() -> None:
    client = RagApiClient()
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"chapter_id": "Chapter_16", "paragraphs": [], "full_text": ""}

    class FakeHttpClient:
        def post(self, path, json):
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    client._client = FakeHttpClient()
    client.get_raw_chapter({"chapter_id": "Chapter_16"})

    assert captured == {
        "path": "/retrieve/raw-chapter",
        "json": {"chapter_id": "Chapter_16"},
    }
