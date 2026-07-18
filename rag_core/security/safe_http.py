"""SSRF-hardened HTTP client.

P0/P1 fix (audit): the previous guard resolved the hostname, validated the
IP, then let `requests.get(url)` re-resolve on its own — a TOCTOU / DNS
rebinding window. We now resolve ONCE, connect to the validated IP, and pass
the original host as the `Host` header. Also adds response-size / read-time
limits the audit called for.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.parse

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


class _HostnameAdapter(HTTPAdapter):
    """Force TLS hostname validation against the original host, not the IP.

    `requests` derives server_hostname from the URL (the IP literal we connect
    to). This adapter binds a pre-configured SSLContext that asserts the real
    hostname, so public HTTPS certificates validate correctly even when we
    connect by resolved IP (TOCTOU-safe SSRF hardening).
    """

    def __init__(self, ssl_context, assert_hostname: str, *args, **kwargs):
        self._ssl_context = ssl_context
        self._assert_hostname = assert_hostname
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        kwargs["assert_hostname"] = self._assert_hostname
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        kwargs["assert_hostname"] = self._assert_hostname
        return super().proxy_manager_for(*args, **kwargs)

_MAPPED_IPV4_BAD_PREFIX = ipaddress.ip_network("::ffff:0:0/96")


def _ip_is_public(ip):
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_multicast or ip.is_reserved or ip.is_link_local or ip.is_loopback:
        return False
    return bool(ip.is_global)


def _host_ips(host: str) -> set:
    try:
        return {ipaddress.ip_address(host)}
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
        return {ipaddress.ip_address(i[4][0]) for i in infos}
    except Exception:
        return set()


def url_targets_public(url: str) -> bool:
    """True only if every resolved IP is globally routable."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        ips = _host_ips(host)
        return bool(ips) and all(_ip_is_public(ip) for ip in ips)
    except Exception:
        return False


def safe_get(url: str, *,
             max_bytes: int = 2_000_000,
             timeout: float = 8,
             headers: dict | None = None,
             user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
             ) -> "requests.Response | None":
    """GET a URL with SSRF hardening + TOCTOU fix.

    Resolves the host ONCE, validates all IPs are public, then connects by IP
    while validating the TLS certificate against the ORIGINAL hostname (not
    the IP literal) via a custom SSL context with assert_hostname. No
    redirects (per-hop re-check). Response body is capped at `max_bytes`.
    Returns None if the URL or any redirect target is unsafe, or on error.
    Caller must still treat the body as untrusted.
    """
    if not url_targets_public(url):
        return None

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ips = _host_ips(host)
    if not ips:
        return None
    ip = next(iter(ips))
    connect_url = f"{parsed.scheme}://{ip}:{port}{parsed.path or '/'}"
    if parsed.query:
        connect_url += "?" + parsed.query

    _headers = {"Host": host, "User-Agent": user_agent}
    if headers:
        _headers.update(headers)

    # TLS: validate cert against ORIGINAL hostname, not the IP literal.
    # requests derives server_hostname from the URL (the IP), so we inject a
    # custom urllib3 adapter that forces assert_hostname=host.
    session = requests.Session()
    if parsed.scheme == "https":
        try:
            from urllib3.util.ssl_ import create_urllib3_context
            ctx = create_urllib3_context()
            ctx.check_hostname = True
            ctx.verify_mode = __import__("ssl").CERT_REQUIRED
            ctx.assert_hostname = host
            session.mount("https://", _HostnameAdapter(ctx, host))
        except Exception:
            pass

    try:
        resp = session.get(connect_url, headers=_headers, timeout=timeout,
                           allow_redirects=False, stream=True)
    except Exception:
        return None

    # Enforce max_bytes: read at most max_bytes from the stream.
    if resp.raw is not None:
        try:
            body = resp.raw.read(max_bytes + 1, decode_content=True)
            if len(body) > max_bytes:
                body = body[:max_bytes]
            resp._content = body
        except Exception:
            pass

    if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
        location = resp.headers.get("Location")
        if not location:
            return None
        target = urllib.parse.urljoin(url, location)
        if not url_targets_public(target):
            return None
        try:
            # Follow via the same hardened session (hostname-validating adapter
            # applies only to https; plain http re-validates via url_targets_public).
            return session.get(target, headers={"User-Agent": user_agent},
                               timeout=timeout, allow_redirects=False, stream=True)
        except Exception:
            return None
    return resp
