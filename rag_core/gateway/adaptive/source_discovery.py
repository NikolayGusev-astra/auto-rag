"""Discover product documentation routes from the local LLM wiki."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import yaml

from rag_core.gateway.config_schema import GatewayConfig
from rag_core.gateway.connectors.confluence_connector import ConfluenceConnector
from rag_core.gateway.secrets import resolve_credential


class SourceDiscovery:
    """Build a product-to-Confluence routing table from wiki product pages."""

    def __init__(self, wiki_path: str | Path, config: GatewayConfig) -> None:
        self._wiki = Path(wiki_path)
        self._config = config

    async def discover(self) -> dict[str, dict[str, str]]:
        connector = self._confluence_connector()
        routing: dict[str, dict[str, str]] = {}
        if connector is not None:
            for page_path in sorted((self._wiki / "rusbitech" / "products").glob("*.md")):
                metadata = _frontmatter(page_path)
                route = _route_from_metadata(metadata)
                if route is None:
                    continue
                children = await connector.child_pages(route["doc_root"])
                route.update(_named_child_pages(children))
                routing[_slug(route["name"])] = route

        destination = _routing_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(routing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return routing

    async def update_routing(self) -> dict[str, dict[str, str]]:
        return await self.discover()

    def _confluence_connector(self) -> ConfluenceConnector | None:
        source = next(
            (item for item in self._config.sources.values() if item.enabled and item.kind == "confluence"),
            None,
        )
        if source is None:
            return None
        base_url = source.extra.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            return None
        try:
            token = resolve_credential(source.credential_ref)
        except KeyError:
            return None
        return ConfluenceConnector(base_url, token, source=source.name)


def _routing_path() -> Path:
    return Path.home() / ".config" / "auto-rag" / "routing.json"


def _frontmatter(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", content, re.DOTALL)
    if not match:
        return {}
    parsed = yaml.safe_load(match.group(1))
    return parsed if isinstance(parsed, dict) else {}


def _route_from_metadata(metadata: dict[str, Any]) -> dict[str, str] | None:
    name = metadata.get("product") or metadata.get("name")
    space = metadata.get("space") or metadata.get("space_key") or metadata.get("confluence_space")
    doc_root = metadata.get("doc_root") or metadata.get("doc_root_page_id")
    if not all(isinstance(value, (str, int, float)) for value in (name, space, doc_root)):
        return None
    route = {"name": str(name), "space": str(space), "doc_root": str(doc_root)}
    if "version" in metadata:
        route["version"] = str(metadata["version"])
    return route


def _named_child_pages(children: list[dict[str, Any]]) -> dict[str, str]:
    pages: dict[str, str] = {}
    for child in children:
        page_id = child.get("id")
        title = str(child.get("title") or "").casefold()
        if page_id is None:
            continue
        if "pmi" in title:
            pages["pmi_page"] = str(page_id)
        elif "\u0440\u0430" in title or re.search(r"\bra\b", title):
            pages["ra_page"] = str(page_id)
    return pages


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
