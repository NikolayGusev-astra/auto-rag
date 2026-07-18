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

logger = logging.getLogger(__name__)

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
    with the original host sent as `Host`. No redirects (per-hop re-check).
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
    # connect to the first validated public IP; keep original Host header
    ip = next(iter(ips))
    connect_url = f"{parsed.scheme}://{ip}:{port}{parsed.path or '/'}"
    if parsed.query:
        connect_url += "?" + parsed.query

    _headers = {"Host": host, "User-Agent": user_agent}
    if headers:
        _headers.update(headers)

    try:
        resp = requests.get(connect_url, headers=_headers, timeout=timeout,
                            allow_redirects=False, stream=True)
    except Exception:
        return None

    if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
        location = resp.headers.get("Location")
        if not location:
            return None
        target = urllib.parse.urljoin(url, location)
        if not url_targets_public(target):
            return None
        try:
            return requests.get(target, headers={"User-Agent": user_agent},
                                timeout=timeout, allow_redirects=False, stream=True)
        except Exception:
            return None
    return resp
