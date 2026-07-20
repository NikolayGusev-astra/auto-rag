"""Learn source priorities from persisted retrieval episodes."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


class DcdLearner:
    """Learns routing policy from persisted episodes."""

    def __init__(self, episodes_path: str | Path, routing_path: str | Path) -> None:
        self._episodes = Path(episodes_path)
        self._routing = Path(routing_path)

    def learn(self) -> dict:
        return self.learn_from_episodes(self._read_episodes())

    def learn_from_episodes(self, episodes: list[dict]) -> dict:
        affinities: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0])
        )
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            query = episode.get("query")
            sources = episode.get("route")
            document_ids = episode.get("document_ids")
            if not isinstance(query, str) or not query.strip():
                continue
            if isinstance(sources, str):
                sources = [sources]
            if not isinstance(sources, (list, tuple)) or not isinstance(document_ids, (list, tuple)):
                continue
            score = _reranker_score(episode)
            for keyword in _tokenize(query):
                for source in sources:
                    if not isinstance(source, str) or not source:
                        continue
                    total, count = affinities[keyword][source]
                    affinities[keyword][source] = [total + score, count + 1]

        if not affinities:
            return {}

        learned = {
            keyword: {
                source: total / count
                for source, (total, count) in sorted(sources.items())
            }
            for keyword, sources in sorted(affinities.items())
        }
        routing = _load_routing(self._routing)
        existing = routing.get("source_affinity")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for keyword, sources in learned.items():
            current = merged.get(keyword)
            updated = dict(current) if isinstance(current, dict) else {}
            updated.update(sources)
            merged[keyword] = updated
        routing["source_affinity"] = merged
        self._routing.parent.mkdir(parents=True, exist_ok=True)
        self._routing.write_text(
            json.dumps(routing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return routing

    def _read_episodes(self) -> list[dict]:
        if not self._episodes.exists():
            return []
        episodes = []
        with self._episodes.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(episode, dict):
                    episodes.append(episode)
        return episodes


def _load_routing(path: Path) -> dict[str, Any]:
    try:
        routing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return routing if isinstance(routing, dict) else {}


def _tokenize(query: str) -> set[str]:
    return set(re.findall(r"\w+", query.casefold(), flags=re.UNICODE))


def _reranker_score(episode: dict) -> float:
    score = episode.get("reranker_score", 1.0)
    return float(score) if isinstance(score, int | float) else 1.0
