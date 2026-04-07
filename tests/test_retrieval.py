from types import SimpleNamespace

from shared.repository import _rerank_summary_score


def test_reranker_prefers_mentioned_but_not_present_for_first_mention_query() -> None:
    query = "任隊長第一次被提到、但人還沒出現，是在哪一段？"

    mentioned_only = SimpleNamespace(
        paragraph_id=1,
        scene="學院東側走廊",
        plot="林妍在任隊長尚未現身前，先從走廊傳聞得知他被撤換。",
        characters=["林妍", "梅子"],
        mentioned_characters=["任隊長"],
        tags=["傳聞"],
        key_events=["林妍在集合前聽見任隊長被撤換。"],
    )
    appears_in_person = SimpleNamespace(
        paragraph_id=2,
        scene="行政樓頂樓",
        plot="任隊長先私下警告林妍不要照單全收報告內容。",
        characters=["林妍", "任隊長"],
        mentioned_characters=["梅子"],
        tags=["警告"],
        key_events=["任隊長私下攔下林妍。"],
    )

    mention_score = _rerank_summary_score(query, mentioned_only, 0.57)
    appearance_score = _rerank_summary_score(query, appears_in_person, 0.575)

    assert mention_score > appearance_score
