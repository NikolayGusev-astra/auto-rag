"""Generic Federated RAG Client — подключение к удалённым RAG сервисам через SSH tunnel или HTTP.

Конфигурация через env vars:
  RAG_FEDERATED_ENABLED=true
  RAG_FEDERATED_SERVERS=server1,server2  # список имён
  RAG_FEDERATED_<name>_HOST=host
  RAG_FEDERATED_<name>_USER=ubuntu
  RAG_FEDERATED_<name>_PORT=22
  RAG_FEDERATED_<name>_REMOTE_PORT=8000
  RAG_FEDERATED_<name>_KEY=~/.ssh/key
  RAG_FEDERATED_<name>_USE_SSH=true

Для конкретных хостов используйте environment variables (RAG_FEDERATED_*).
"""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os
import subprocess
import time
import warnings
from typing import Any, Optional
from dataclasses import dataclass, field

import aiohttp


logger = logging.getLogger(__name__)

FEDERATION_EXPERIMENTAL = True
MAX_FEDERATION_HOPS = 3
_federation_hop_count: ContextVar[int] = ContextVar("federation_hop_count", default=0)


@contextmanager
def federation_hop_context(hop_count: int):
    """Propagate an inbound federation hop count to downstream requests."""
    token = _federation_hop_count.set(hop_count)
    try:
        yield
    finally:
        _federation_hop_count.reset(token)


@dataclass
class FederatedServerConfig:
    """Конфиг одного удалённого RAG сервера."""
    name: str
    host: str
    user: str = "ubuntu"
    port: int = 22
    remote_port: int = 8000
    key_path: Optional[str] = None
    use_ssh: bool = True
    endpoint: str = "/rag/search"
    timeout: int = 30
    # Audit S6: use HTTPS for the direct (non-SSH) mode so the API key is
    # not sent in cleartext. SSH-tunnelled and localhost paths stay http.
    use_tls: bool = True
    
    # Runtime fields
    _tunnel_proc: Optional[subprocess.Popen] = field(default=None, init=False)
    local_port: Optional[int] = field(default=None, init=False)
    accept_new_host: bool = False  # автоматически добавлять новые хосты в known_hosts


class _ServerHealth:
    """Health state для одного сервера."""
    def __init__(self) -> None:
        self.consecutive_failures: int = 0
        self.last_failure_ts: float = 0
        self.last_success_ts: float = 0
        self.cooldown_until: float = 0

    @property
    def is_healthy(self) -> bool:
        return time.time() >= self.cooldown_until


class FederatedRAGClient:
    """Клиент для параллельного опроса нескольких удалённых RAG серверов."""
    
    def __init__(self, configs: list[FederatedServerConfig]):
        self.configs = {c.name: c for c in configs}
        self._ssh_config_written = False
        self._health: dict[str, _ServerHealth] = {name: _ServerHealth() for name in self.configs}
        self._domain_map: dict[str, list[str]] = {}
        self._server_capabilities: dict[str, dict] = {}
        self._domain_map_ts = 0.0
        self._domain_map_ttl = 300
    
    @classmethod
    def from_env(cls) -> "FederatedRAGClient":
        """Создать клиент из environment variables.
        
        Env format:
        RAG_FEDERATED_ENABLED=true
        RAG_FEDERATED_SERVERS=server1,server2
        RAG_FEDERATED_<name>_HOST=host
        RAG_FEDERATED_<name>_USER=ubuntu
        RAG_FEDERATED_<name>_PORT=22
        RAG_FEDERATED_<name>_REMOTE_PORT=8000
        RAG_FEDERATED_<name>_KEY=~/.ssh/key
        RAG_FEDERATED_<name>_USE_SSH=true
        RAG_FEDERATED_<name>_ENDPOINT=/rag/search
        RAG_FEDERATED_<name>_TIMEOUT=30
        """
        if not os.getenv("RAG_FEDERATED_ENABLED", "false").lower() == "true":
            return cls([])
        
        server_names = os.getenv("RAG_FEDERATED_SERVERS", "").split(",")
        configs = []
        
        for name in server_names:
            name = name.strip().lower()
            if not name:
                continue
            prefix = f"RAG_FEDERATED_{name.upper()}_"
            
            host = os.getenv(f"{prefix}HOST")
            if not host:
                continue
                
            key = os.getenv(f"{prefix}KEY")
            if key:
                key = os.path.expanduser(key)
            
            config = FederatedServerConfig(
                name=name,
                host=host,
                user=os.getenv(f"{prefix}USER", "ubuntu"),
                port=int(os.getenv(f"{prefix}PORT", "22")),
                remote_port=int(os.getenv(f"{prefix}REMOTE_PORT", "8000")),
                key_path=key,
                use_ssh=os.getenv(f"{prefix}USE_SSH", "true").lower() == "true",
                endpoint=os.getenv(f"{prefix}ENDPOINT", "/rag/search"),
                timeout=int(os.getenv(f"{prefix}TIMEOUT", "30")),
            )
            configs.append(config)
        
        return cls(configs)
    
    def _write_ssh_config(self):
        """Написать временный SSH config для туннелей."""
        if self._ssh_config_written:
            return
        
        ssh_dir = os.path.expanduser("~/.ssh")
        os.makedirs(ssh_dir, exist_ok=True)
        config_path = os.path.join(ssh_dir, "config.federated_rag")
        
        lines = ["# Federated RAG SSH config - auto-generated\n"]
        for config in self.configs.values():
            if not config.use_ssh:
                continue
            lines.append(f"Host federated-{config.name}")
            lines.append(f"    HostName {config.host}")
            lines.append(f"    User {config.user}")
            lines.append(f"    Port {config.port}")
            if config.key_path and os.path.exists(config.key_path):
                lines.append(f"    IdentityFile {config.key_path}")
            if config.accept_new_host:
                lines.append("    StrictHostKeyChecking accept-new")
            # По умолчанию ssh требует ручного подтверждения known_hosts
            lines.append("    ServerAliveInterval 30")
            lines.append("    ServerAliveCountMax 3")
            lines.append("    ExitOnForwardFailure yes")
            lines.append("    ConnectTimeout 10")
            lines.append("")
        
        with open(config_path, "w") as f:
            f.write("\n".join(lines))
        
        os.chmod(config_path, 0o600)
        os.environ["SSH_CONFIG"] = config_path
        self._ssh_config_written = True
    
    async def _ensure_tunnel(self, config: FederatedServerConfig) -> int:
        """Поднять SSH tunnel если нужно."""
        if config.use_ssh:
            if config._tunnel_proc and config._tunnel_proc.poll() is None:
                return config.local_port
        else:
            # No tunnel needed for direct HTTP
            return config.remote_port
        
        import random
        config.local_port = random.randint(18000, 19000)
        
        self._write_ssh_config()
        ssh_config = os.environ.get("SSH_CONFIG", os.path.expanduser("~/.ssh/config"))
        
        cmd = [
            "ssh", "-F", ssh_config, "-N",
            "-L", f"{config.local_port}:localhost:{config.remote_port}",
            f"federated-{config.name}",
        ]
        
        config._tunnel_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Wait for tunnel
        await asyncio.sleep(2)
        
        # Verify tunnel
        try:
            import socket
            sock = socket.create_connection(("localhost", config.local_port), timeout=3)
            sock.close()
        except Exception:
            if config._tunnel_proc:
                config._tunnel_proc.terminate()
            raise RuntimeError(f"SSH tunnel failed for {config.name}")
        
        return config.local_port
    
    def _get_session(self, config: FederatedServerConfig) -> aiohttp.ClientSession:
        """Create a request-scoped session on the current event loop."""
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout))
    
    def _build_url(self, config: FederatedServerConfig) -> str:
        """Построить URL для запроса.

        SSH tunnel and localhost paths are already protected, so http:// is
        fine there. For the direct (non-SSH) mode S6 requires HTTPS so the
        API key is not sent in cleartext.
        """
        if config.use_ssh and config.local_port:
            return f"http://localhost:{config.local_port}{config.endpoint}"
        scheme = "https" if config.use_tls else "http"
        return f"{scheme}://{config.host}:{config.remote_port}{config.endpoint}"
    
    async def query(self, name: str, query: str, max_results: int = 5) -> list[dict]:
        """Запрос к одному удалённому RAG серверу с circuit breaker."""
        config = self.configs.get(name)
        if not config:
            return [{"text": f"Unknown federated server: {name}", "source": name, "score": 0, "is_error": True}]

        health = self._health.get(name)
        if health and not health.is_healthy:
            return [{"text": f"Server {name} in cooldown (circuit breaker)",
                     "source": name, "score": 0, "is_error": True}]

        try:
            result = await self._do_query(config, name, query, max_results)
            if health:
                health.consecutive_failures = 0
                health.last_success_ts = time.time()
            return result
        except Exception as e:
            if health:
                health.consecutive_failures += 1
                health.last_failure_ts = time.time()
                if health.consecutive_failures >= 3:
                    health.cooldown_until = time.time() + 300
            return [{"text": f"Federated RAG {name} error: {e}", "source": name, "score": 0, "is_error": True}]

    async def _do_query(self, config: FederatedServerConfig, name: str, query: str, max_results: int) -> list[dict]:
        """Исходная логика query с retry на transient errors."""
        if config.use_ssh:
            await self._ensure_tunnel(config)

        session = self._get_session(config)
        url = self._build_url(config)
        headers = {}
        # S9: each forwarded request increments the bounded hop count.  The
        # receiving endpoint rejects values above MAX_FEDERATION_HOPS.
        headers["X-Federation-Hop"] = str(_federation_hop_count.get() + 1)
        api_key = os.getenv("RAG_FEDERATED_API_KEY")
        if api_key:
            headers["X-API-Key"] = api_key

        payload = {"query": query, "max_results": max_results}
        max_retries = 2
        last_error = None

        try:
            for attempt in range(max_retries + 1):
                try:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            chunks = data.get("chunks", data.get("results", []))
                            normalized = []
                            for c in chunks[:max_results]:
                                if isinstance(c, dict):
                                    text = c.get("text", c.get("content", c.get("body", "")))
                                    if text:
                                        normalized.append({
                                            "text": text[:800],
                                            "score": c.get("score", 0.5),
                                            "source": c.get("source", name),
                                            "metadata": c.get("metadata", {}),
                                        })
                            return normalized
                        elif resp.status >= 500:
                            last_error = RuntimeError(f"HTTP {resp.status}")
                            if attempt < max_retries:
                                await asyncio.sleep(0.5 * (attempt + 1))
                                continue
                            text = await resp.text()
                            return [{"text": f"Remote RAG {name} error {resp.status}: {text}", "source": name, "score": 0, "is_error": True}]
                        else:
                            text = await resp.text()
                            return [{"text": f"Remote RAG {name} error {resp.status}: {text}", "source": name, "score": 0, "is_error": True}]
                except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    raise

            raise last_error or RuntimeError("Unknown error after retries")
        finally:
            await session.close()
    
    async def query_all(self, query: str, max_results: int = 5) -> dict[str, list[dict]]:
        """Параллельный запрос ко всем настроенным серверам."""
        tasks = [
            self.query(name, query, max_results)
            for name in self.configs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out = {}
        for name, result in zip(self.configs.keys(), results):
            if isinstance(result, Exception):
                out[name] = [{"text": f"Error: {result}", "source": name, "score": 0}]
            else:
                out[name] = result
        return out

    async def _fetch_domains(self, name: str) -> dict | None:
        """Получить список доменов + capabilities с сервера."""
        config = self.configs.get(name)
        if not config:
            return None
        try:
            if config.use_ssh:
                await self._ensure_tunnel(config)
            session = self._get_session(config)
            url = self._build_url(config).replace("/rag/search", "/domains")
            headers = {}
            api_key = os.getenv("RAG_FEDERATED_API_KEY")
            if api_key:
                headers["X-API-Key"] = api_key

            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()
            finally:
                await session.close()
        except Exception:
            return None
        return None

    async def _load_domain_map(self, force: bool = False) -> None:
        """Опросить /domains на каждом сервере."""
        if not force and self._domain_map and time.time() - self._domain_map_ts < self._domain_map_ttl:
            return

        new_map: dict[str, list[str]] = {}
        self._server_capabilities = {}

        tasks = [self._fetch_domains(name) for name in self.configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(self.configs.keys(), results):
            if isinstance(result, Exception) or not result:
                continue
            for domain in result.get("domains", []):
                new_map.setdefault(domain, []).append(name)
            self._server_capabilities[name] = result.get("capabilities", {})

        self._domain_map = new_map
        self._domain_map_ts = time.time()

    async def query_routed(self, query: str, domain: str, max_results: int = 5) -> dict[str, list[dict]]:
        """Запрос только на серверы, у которых есть нужный домен."""
        await self._load_domain_map()
        target_servers = []
        for name, domains in self._domain_map.items():
            if domain in domains and name in self.configs:
                target_servers.append(name)

        if not target_servers:
            return await self.query_all(query, max_results)

        tasks = [self.query(name, query, max_results) for name in target_servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out = {}
        for name, result in zip(target_servers, results):
            if isinstance(result, Exception):
                out[name] = [{"text": f"Error: {result}", "source": name, "score": 0}]
            else:
                out[name] = result
        return out

    async def close(self):
        """Закрыть все туннели и сессии."""
        for config in self.configs.values():
            if config._tunnel_proc:
                config._tunnel_proc.terminate()
                config._tunnel_proc = None


# ── Singleton client (persistent tunnels) ──────────────────────────
_CLIENT: "FederatedRAGClient | None" = None
_CLIENT_LOCK = asyncio.Lock()


async def get_federated_client() -> "FederatedRAGClient":
    """Singleton federated client. Tunnels live for the process lifetime."""
    global _CLIENT
    if _CLIENT is not None and _CLIENT.configs:
        return _CLIENT
    async with _CLIENT_LOCK:
        if _CLIENT is None or not _CLIENT.configs:
            _CLIENT = FederatedRAGClient.from_env()
    return _CLIENT


async def shutdown_federated() -> None:
    """Закрыть singleton client."""
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.close()
        _CLIENT = None


# ── Helper для интеграции в rag_async ────────────────────────────
# Использует persistent singleton клиент, чтобы не разрывать SSH-туннели
# между последовательными запросами. Закрывается через shutdown_federated().
async def query_federated_servers(query: str, max_results: int = 3, domain: str = "") -> dict[str, list[dict]]:
    """Удобная функция для вызова из rag_async."""
    warnings.warn(
        "Federation is an experimental extension and is not part of the gateway reference path.",
        RuntimeWarning,
        stacklevel=2,
    )
    client = await get_federated_client()
    if not client.configs:
        return {}
    try:
        if domain:
            return await client.query_routed(query, domain, max_results)
        return await client.query_all(query, max_results)
    except Exception as exc:
        logger.exception("Federated query failed")
        return {
            "federation": [{
                "text": f"Federated query error: {exc}",
                "source": "federation",
                "score": 0,
                "is_error": True,
            }]
        }
