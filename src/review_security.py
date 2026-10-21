"""Same-origin session protection for the small, trusted-device review UIs."""

from __future__ import annotations

import hmac
import ipaddress
import json
import re
import secrets
import socket
from pathlib import Path
from urllib.parse import urlsplit

SESSION_ROUTE = "/api/review-session"
SCRIPT_ROUTE = "/api/review-security.js"
TOKEN_HEADER = "X-Review-Token"


def normalized_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Review origins must be explicit http(s) origins without paths.")
    port = parsed.port
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != (443 if parsed.scheme == "https" else 80):
        host += f":{port}"
    return f"{parsed.scheme}://{host}"


def configure_review_server(server, allowed_origins=(), host: str | None = None) -> None:
    """Allow the bound address plus explicitly configured reverse-proxy origins."""
    bound_host, port = server.server_address[:2]
    hosts = {bound_host}
    try:
        loopback = ipaddress.ip_address(bound_host).is_loopback
    except ValueError:
        loopback = bound_host.lower() == "localhost"
    if loopback:
        hosts.update({"127.0.0.1", "localhost", "::1"})
    if host and host not in {"0.0.0.0", "::"}:
        hosts.add(host)
    origins = {normalized_origin(value) for value in allowed_origins}
    for name in hosts - {"0.0.0.0", "::"}:
        address = f"[{name}]" if ":" in name else name
        origins.add(normalized_origin(f"http://{address}:{port}"))
    if not origins:
        raise ValueError("Wildcard binding requires an explicit --allowed-origin.")
    server.review_origins = frozenset(origins)
    server.review_hosts = frozenset(urlsplit(origin).netloc for origin in origins)
    server.review_token = secrets.token_urlsafe(32)


def review_html(raw: bytes) -> bytes:
    """Load the request helper before existing scripts, including external UIs."""
    tag = f'<script src="{SCRIPT_ROUTE}"></script>'.encode()
    if tag in raw:
        return raw
    head = re.search(br"<head\b[^>]*>", raw, re.IGNORECASE)
    if head:
        marker = head.end()
        return raw[:marker] + tag + raw[marker:]
    # Documents without an explicit head still retain their leading doctype.
    doctype = re.match(br"\s*<!doctype\b[^>]*>", raw, re.IGNORECASE)
    marker = doctype.end() if doctype else 0
    return raw[:marker] + tag + raw[marker:]


class ReviewSecurityMixin:
    """Reject browser mutation attempts before routing or consuming their bodies."""

    def _review_reply(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _review_deny(self, status: int, message: str) -> bool:
        self.close_connection = True  # Never reinterpret an unread rejected body.
        self._review_reply(status, json.dumps({"ok": False, "error": message}).encode(), "application/json")
        # Half-close the response first: closing an unread POST on Windows can
        # otherwise reset the connection before the client receives the denial.
        self.wfile.flush()
        try:
            self.connection.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        return False

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        if not hasattr(self.server, "review_token"):
            return self._review_deny(503, "Review security is not configured.")
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0].lower() not in self.server.review_hosts:
            return self._review_deny(403, "Untrusted review Host.")
        origins = self.headers.get_all("Origin", [])
        if origins:
            try:
                trusted = len(origins) == 1 and normalized_origin(origins[0]) in self.server.review_origins
            except ValueError:
                trusted = False
            if not trusted:
                return self._review_deny(403, "Untrusted review Origin.")
        if self.command in {"POST", "PUT", "PATCH", "DELETE"}:
            supplied = self.headers.get_all(TOKEN_HEADER, [])
            if len(supplied) != 1 or not hmac.compare_digest(supplied[0].encode(), self.server.review_token.encode()):
                return self._review_deny(403, "Missing or invalid review session token; reload the review page.")
            types = self.headers.get_all("Content-Type", [])
            if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json":
                return self._review_deny(415, "Review mutations require application/json.")
        route = urlsplit(getattr(self, "path", "")).path
        if self.command == "GET" and route == SESSION_ROUTE:
            self._review_reply(200, json.dumps({"token": self.server.review_token}).encode(), "application/json")
            return False
        if self.command == "GET" and route == SCRIPT_ROUTE:
            self._review_reply(200, Path(__file__).with_suffix(".js").read_bytes(), "application/javascript")
            return False
        return True

    def end_headers(self) -> None:
        route = urlsplit(getattr(self, "path", "")).path
        if route.endswith(("/", ".html")):
            self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        super().end_headers()
