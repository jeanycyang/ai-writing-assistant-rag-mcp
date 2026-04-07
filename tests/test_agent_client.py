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
