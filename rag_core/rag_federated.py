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

Для конкретных хостов используйте rustbitech_servers.json (не в коде).
"""

import asyncio
import os
import subprocess
from typing import Any, Optional
from dataclasses import dataclass, field

import aiohttp


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
    
    # Runtime fields
    _tunnel_proc: Optional[subprocess.Popen] = field(default=None, init=False)
    local_port: Optional[int] = field(default=None, init=False)
    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False)


class FederatedRAGClient:
    """Клиент для параллельного опроса нескольких удалённых RAG серверов."""
    
    def __init__(self, configs: list[FederatedServerConfig]):
        self.configs = {c.name: c for c in configs}
        self._ssh_config_written = False
    
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
            lines.append("    StrictHostKeyChecking no")
            lines.append("    UserKnownHostsFile /dev/null")
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
        """Получить или создать aiohttp сессию."""
        if config._session is None or config._session.closed:
            config._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            )
        return config._session
    
    def _build_url(self, config: FederatedServerConfig) -> str:
        """Построить URL для запроса."""
        if config.use_ssh and config.local_port:
            return f"http://localhost:{config.local_port}{config.endpoint}"
        else:
            return f"http://{config.host}:{config.remote_port}{config.endpoint}"
    
    async def query(self, name: str, query: str, max_results: int = 5) -> list[dict]:
        """Запрос к одному удалённому RAG серверу."""
        config = self.configs.get(name)
        if not config:
            return [{"text": f"Unknown federated server: {name}", "source": name, "score": 0}]
        
        try:
            # Ensure tunnel if needed
            if config.use_ssh:
                await self._ensure_tunnel(config)
            
            session = self._get_session(config)
            url = self._build_url(config)
            
            async with session.post(url, json={
                "query": query,
                "max_results": max_results,
            }) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return [{"text": f"Remote RAG {name} error {resp.status}: {text}", "source": name, "score": 0}]
                data = await resp.json()
                chunks = data.get("chunks", data.get("results", []))
                
                # Normalize chunk format
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
                
        except Exception as e:
            return [{"text": f"Federated RAG {name} error: {e}", "source": name, "score": 0}]
    
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
    
    async def close(self):
        """Закрыть все туннели и сессии."""
        for config in self.configs.values():
            if config._tunnel_proc:
                config._tunnel_proc.terminate()
                config._tunnel_proc = None
            if config._session and not config._session.closed:
                await config._session.close()
                config._session = None


# ── Helper для интеграции в rag_async ──

async def query_federated_servers(query: str, max_results: int = 3) -> dict[str, list[dict]]:
    """Удобная функция для вызова из rag_async."""
    client = FederatedRAGClient.from_env()
    if not client.configs:
        return {}
    try:
        return await client.query_all(query, max_results)
    finally:
        await client.close()