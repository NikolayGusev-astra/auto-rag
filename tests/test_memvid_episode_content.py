from unittest.mock import MagicMock, patch

from rag_core import rag_async
from rag_trace import RagTrace


def test_episode_answer_uses_chunks_not_trace():
    result = {
        "source": "zvec",
        "trace": "ZVec(0.91)",
        "chunks": [
            {"text": "Полезный факт один", "source": "wiki"},
            {"content": "Полезный факт два", "source": "docs"},
        ],
    }
    assert rag_async._episode_answer(result) == "Полезный факт один\n\nПолезный факт два"


def test_record_episode_stores_content_and_sources():
    result = {
        "source": "zvec",
        "trace": "ZVec(0.91)",
        "chunks": [{"text": "Настоящий ответ из индекса", "source": "wiki"}],
    }
    memory = MagicMock()
    memory.active = True
    trace = RagTrace("тест", "database", "database")

    with patch.object(rag_async, "_MEMVID_AVAILABLE", True), \
         patch.object(rag_async, "_get_memory", return_value=memory):
        rag_async._record_episode(result, "тестовый запрос", "database", trace)

    episode = memory.record.call_args.args[0]
    assert episode.answer == "Настоящий ответ из индекса"
    assert episode.answer != result["trace"]
    # new episode_writer format: sources carry trust flag (poisoning guard)
    assert episode.sources == [{"source": "wiki", "trusted": True}]
